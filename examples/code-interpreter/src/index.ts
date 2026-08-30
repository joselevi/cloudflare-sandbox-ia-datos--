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

const isValidObjectKey = (objectKey: unknown): objectKey is string => {
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
        .map((r) => r.text || r.html || JSON.stringify(r))
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

      output += `Error: ${result.logs.stderr.join('\n')}`;
    }

    return result.error
      ? `Error: ${result.error}`
      : output || 'Code executed successfully';
  } finally {
    if (bucketMounted) {
      try {
        await sandbox.unmountBucket(MOUNT_PATH);
      } catch (error) {
        console.error('R2 bucket unmount failed:', error);
      }
    }
  }
}

async function handleAIRequest(
  input: string,
  env: Env
): Promise<string> {
  const workersai = createWorkersAI({
    binding: env.AI
  });

  const result = await generateText({
    model: workersai(MODEL),
    messages: [
      {
        role: 'user',
        content: input
      }
    ],
    tools: {
      execute_python: tool({
        description: 'Execute Python code and return the output',
        inputSchema: z.object({
          code: z.string().describe(
            'The Python code to execute'
          )
        }),
        execute: async ({ code }) => {
          return executePythonCode(env, code);
        }
      })
    },
    stopWhen: stepCountIs(5)
  });

  return result.text || 'No response generated';
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

      let requestInput = input;

      if (!requestInput && objectKey) {
        requestInput = [
          prompt || '',
          `El archivo está disponible en ${MOUNT_PATH}/${objectKey}.`,
          'Utiliza execute_python para ejecutar una prueba sobre ese archivo.'
        ].join('\n\n');
      }

      if (!requestInput) {
        return Response.json(
          {
            error: 'Missing input or objectKey field'
          },
          {
            status: 400
          }
        );
      }

      const output = await handleAIRequest(
        requestInput,
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
