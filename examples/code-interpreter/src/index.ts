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

  console.log('[Worker] executePythonCode START');

  try {
    console.log('[Worker] mountBucket START');

    await sandbox.mountBucket(
      'IA_DATOS_BUCKET',
      MOUNT_PATH,
      {
        readOnly: true
      }
    );

    bucketMounted = true;

    console.log('[Worker] mountBucket END');

    console.log('[Worker] createCodeContext START');

    const pythonCtx = await sandbox.createCodeContext({
      language: 'python'
    });

    console.log('[Worker] createCodeContext END');

    console.log('[Worker] runCode START');

    const result = await sandbox.runCode(code, {
      context: pythonCtx
    });

    console.log('[Worker] runCode END', {
      hasError: Boolean(result.error),
      results: result.results?.length || 0,
      stdout: result.logs?.stdout?.length || 0,
      stderr: result.logs?.stderr?.length || 0
    });

    if (
      result.error ||
      result.logs?.stderr?.length
    ) {
      const pythonError = [
        result.error
          ? String(result.error)
          : '',
        result.logs?.stderr?.length
          ? result.logs.stderr.join('\n')
          : ''
      ]
        .filter(Boolean)
        .join('\n');

      throw new Error(
        pythonError || 'Python execution failed'
      );
    }

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

    if (result.logs?.stdout?.length) {
      return result.logs.stdout.join('\n');
    }

    return 'Code executed successfully';
  } catch (error) {
    console.error('[Worker] executePythonCode ERROR', {
      message: error instanceof Error
        ? error.message
        : 'Unknown error'
    });

    throw error;
  } finally {
    if (bucketMounted) {
      console.log('[Worker] unmountBucket START');

      try {
        await sandbox.unmountBucket(MOUNT_PATH);

        console.log('[Worker] unmountBucket END');
      } catch (error) {
        console.error(
          '[Worker] unmountBucket ERROR',
          {
            message: error instanceof Error
              ? error.message
              : 'Unknown error'
          }
        );
      }
    }

    console.log('[Worker] executePythonCode END');
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
        'Ejecutá execute_python una sola vez.',
        'Leé el CSV completo usando pandas.',
        'El CSV utiliza separador punto y coma (;).',
        'Usá dtype=str para conservar los valores.',
        'No hagas prints intermedios.',
        'El código Python debe imprimir únicamente el JSON final.',
        'Después devolvé únicamente ese JSON válido.'
      ].join('\n')
    : [
        'Procesá la instrucción recibida.',
        'Usá execute_python si corresponde.',
        'Después devolvé la respuesta final.'
      ].join('\n');

  console.log('[Worker] generateText START', {
    hasCsvPath: Boolean(csvPath),
    promptLength: prompt.length
  });

  try {
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
          description: [
            'Ejecuta Python dentro del Sandbox.',
            'Debe leer el CSV completo.',
            'Debe usar el separador ;.',
            'Debe devolver únicamente el JSON final.'
          ].join(' '),
          inputSchema: z.object({
            code: z.string().describe(
              'Código Python completo para ejecutar el análisis'
            )
          }),
          execute: async ({ code }) => {
            console.log(
              '[Worker] execute_python TOOL START'
            );

            const output = await executePythonCode(
              env,
              code
            );

            console.log(
              '[Worker] execute_python TOOL END',
              {
                outputLength: output.length
              }
            );

            return output;
          }
        })
      },
      maxOutputTokens: 8192,
      stopWhen: stepCountIs(2)
    });

    const toolCalls = (result.steps || [])
      .reduce((total, step) => {
        return total + (step.toolCalls?.length || 0);
      }, 0);

    const toolResults = (result.steps || [])
      .reduce((total, step) => {
        return total + (step.toolResults?.length || 0);
      }, 0);

    const finalText = result.text?.trim() || '';

    console.log('[Worker] generateText END', {
      finishReason: result.finishReason || 'unknown',
      steps: result.steps?.length || 0,
      toolCalls,
      toolResults,
      responseLength: finalText.length
    });

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
  } catch (error) {
    console.error('[Worker] generateText ERROR', {
      message: error instanceof Error
        ? error.message
        : 'Unknown error'
    });

    throw error;
  }
}

export default {
  async fetch(
    request: Request,
    env: Env
  ): Promise<Response> {
    const url = new URL(request.url);

    console.log('[Worker] REQUEST START', {
      method: request.method,
      pathname: url.pathname
    });

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
      console.error('[Worker] AUTH ERROR');

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
      console.error('[Worker] ROUTE ERROR', {
        method: request.method,
        pathname: url.pathname
      });

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

      console.log('[Worker] BODY RECEIVED', {
        hasInput: Boolean(input),
        hasObjectKey: Boolean(objectKey),
        hasPrompt: Boolean(prompt)
      });

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

      console.log('[Worker] REQUEST VALIDATED', {
        hasCsvPath: Boolean(csvPath),
        promptLength: requestPrompt.length
      });

      const output = await handleAIRequest(
        requestPrompt,
        csvPath,
        env
      );

      console.log('[Worker] REQUEST END', {
        outputLength: output.length
      });

      return Response.json({
        output
      });
    } catch (error) {
      console.error('[Worker] REQUEST ERROR', {
        message: error instanceof Error
          ? error.message
          : 'Internal Server Error'
      });

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
