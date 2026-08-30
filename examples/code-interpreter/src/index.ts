import {
  ContainerProxy,
  getSandbox
} from '@cloudflare/sandbox';
import {
  generateText,
  stepCountIs,
  tool
} from 'ai';
import { createWorkersAI } from 'workers-ai-provider';
import { z } from 'zod';

export { ContainerProxy };
export { Sandbox } from '@cloudflare/sandbox';

const API_PATH = '/run';
const MODEL = '@cf/openai/gpt-oss-120b' as const;
const MOUNT_PATH = '/mnt/r2';

type WorkerRequest = {
  input?: string;
  objectKey?: string;
  prompt?: string;
};

const isValidObjectKey = (
  objectKey: unknown
): objectKey is string => {
  return (
    typeof objectKey === 'string' &&
    /^analisis\/[0-9a-f-]{36}\.csv$/i.test(objectKey)
  );
};

async function executePythonCode(
  env: Env,
  code: string
): Promise<string> {
  const sandboxId = env.Sandbox.idFromName('default');

  const sandbox = getSandbox(
    env.Sandbox,
    sandboxId.toString().slice(0, 63)
  );

  let bucketMounted = false;

  try {
    await sandbox.mountBucket(
      'IA_DATOS_BUCKET',
      MOUNT_PATH,
      {
        readOnly: true
      }
    );

    bucketMounted = true;

    const pythonCtx = await sandbox.createCodeContext({
      language: 'python'
    });

    const result = await sandbox.runCode(code, {
      context: pythonCtx
    });

    if (result.results?.length) {
      const outputs = result.results
        .map((item) => {
          return item.text ||
            item.html ||
            JSON.stringify(item);
        })
        .filter(Boolean);

      if (outputs.length) {
        return outputs.join('\n');
      }
    }

    let output = '';

    if (result.logs?.stdout?.length) {
      output = result.logs.stdout.join('\n');
    }

    if (result.logs?.stderr?.length) {
      if (output) {
        output += '\n';
      }

      output += `Error: ${
        result.logs.stderr.join('\n')
      }`;
    }

    if (result.error) {
      return `Error: ${result.error}`;
    }

    return output || 'Code executed successfully';
  } finally {
    if (bucketMounted) {
      try {
        await sandbox.unmountBucket(MOUNT_PATH);
      } catch (error) {
        console.error(
          'R2 bucket unmount failed:',
          error
        );
      }
    }
  }
}

async function handleAIRequest(
  prompt: string,
  csvPath: string | undefined,
  env: Env
): Promise<string> {
  const workersai = createWorkersAI({
    binding: env.AI
  });

  const runtimeContext = csvPath
    ? [
        `CSV disponible en ${csvPath}.`,
        'Ejecutá execute_python inmediatamente.',
        'Procesá el CSV completo.',
        'Después devolvé únicamente JSON válido.'
      ].join('\n')
    : [
        'Procesá la instrucción recibida.',
        'Usá execute_python si corresponde.',
        'Después devolvé la respuesta final.'
      ].join('\n');

  const result = await generateText({
    model: workersai(MODEL),
    system: runtimeContext,
    messages: [
      {
        role: 'user',
        content: prompt
      }
    ],
    tools: {
      execute_python: tool({
        description:
          'Ejecuta Python dentro del Sandbox para procesar el CSV completo.',
        inputSchema: z.object({
          code: z.string().describe(
            'Código Python completo para ejecutar el análisis'
          )
        }),
        execute: async ({ code }) => {
          return executePythonCode(env, code);
        }
      })
    },
    maxOutputTokens: 16384,
    stopWhen: stepCountIs(8)
  });

  const finalText = result.text?.trim() || '';

  const toolCalls = (result.steps || [])
    .reduce((total, step) => {
      return total + (step.toolCalls?.length || 0);
    }, 0);

  const toolResults = (result.steps || [])
    .reduce((total, step) => {
      return total + (step.toolResults?.length || 0);
    }, 0);

  if (!finalText) {
    throw new Error(
      [
        'Workers AI no generó respuesta final.',
        `finishReason=${result.finishReason || 'unknown'}`,
        `steps=${result.steps?.length || 0}`,
        `toolCalls=${toolCalls}`,
        `toolResults=${toolResults}`
      ].join(' ')
    );
  }

  if (csvPath) {
    try {
      JSON.parse(finalText);
    } catch {
      throw new Error(
        'Workers AI devolvió una respuesta que no es JSON válido.'
      );
    }
  }

  return finalText;
}

export default {
  async fetch(
    request: Request,
    env: Env
  ): Promise<Response> {
    const url = new URL(request.url);

    const workerToken = (
      env as Env & {
        CLOUDFLARE_IA_DATOS_WORKER_TOKEN?: string;
      }
    ).CLOUDFLARE_IA_DATOS_WORKER_TOKEN;

    const authorization =
      request.headers.get('Authorization');

    const expectedAuthorization = workerToken
      ? `Bearer ${workerToken}`
      : null;

    if (
      !expectedAuthorization ||
      authorization !== expectedAuthorization
    ) {
      return Response.json(
        {
          error: 'Unauthorized'
        },
        {
          status: 401
        }
      );
    }

    if (
      url.pathname !== API_PATH ||
      request.method !== 'POST'
    ) {
      return new Response('Not Found', {
        status: 404
      });
    }

    try {
      const body = await request.json<WorkerRequest>();

      if (!body || typeof body !== 'object') {
        return Response.json(
          {
            error: 'Invalid request body'
          },
          {
            status: 400
          }
        );
      }

      const {
        input,
        objectKey,
        prompt
      } = body;

      if (
        objectKey !== undefined &&
        !isValidObjectKey(objectKey)
      ) {
        return Response.json(
          {
            error: 'Invalid objectKey'
          },
          {
            status: 400
          }
        );
      }

      const usingObjectKey =
        typeof objectKey === 'string';

      if (usingObjectKey && !prompt) {
        return Response.json(
          {
            error: 'Missing prompt field'
          },
          {
            status: 400
          }
        );
      }

      const requestPrompt =
        usingObjectKey
          ? prompt
          : input;

      if (!requestPrompt) {
        return Response.json(
          {
            error: 'Missing input or objectKey field'
          },
          {
            status: 400
          }
        );
      }

      const csvPath = usingObjectKey
        ? `${MOUNT_PATH}/${objectKey}`
        : undefined;

      const output = await handleAIRequest(
        requestPrompt,
        csvPath,
        env
      );

      return Response.json({
        output
      });
    } catch (error) {
      console.error('Request failed:', error);

      const message =
        error instanceof Error
          ? error.message
          : 'Internal Server Error';

      return Response.json(
        {
          error: message
        },
        {
          status: 500
        }
      );
    }
  }
} satisfies ExportedHandler<Env>;
