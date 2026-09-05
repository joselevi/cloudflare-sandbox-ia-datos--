import { ContainerProxy, getSandbox } from '@cloudflare/sandbox';
import { generateText, tool } from 'ai';
import { createWorkersAI } from 'workers-ai-provider';
import { z } from 'zod';

export { ContainerProxy };
export { Sandbox } from '@cloudflare/sandbox';

const API_PATH = '/run';
const MODEL = '@cf/openai/gpt-oss-120b' as const;
const R2_BINDING = 'IA_DATOS_BUCKET';
const R2_MOUNT_PATH = '/mnt/r2';

type SetStage = (stage: string) => void;

function formatPythonResult(result: {
  results?: Array<{
    text?: string;
    html?: string;
  }>;
  logs?: {
    stdout?: string[];
    stderr?: string[];
  };
  error?: unknown;
}): string {
  if (result.results?.length) {
    const outputs = result.results
      .map((item) => item.text || item.html || JSON.stringify(item))
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
    ? `Error: ${String(result.error)}`
    : output || 'Código ejecutado correctamente.';
}

async function handleAIRequest(
  prompt: string,
  objectKey: string,
  env: Env,
  setStage: SetStage
): Promise<string> {
  const sandboxId = env.Sandbox.idFromName('default');

  const sandbox = getSandbox(
    env.Sandbox,
    sandboxId.toString().slice(0, 63)
  );

  const csvPath = `${R2_MOUNT_PATH}/${objectKey}`;

  setStage('sandbox_mount_r2');

  await sandbox.mountBucket(
    R2_BINDING,
    R2_MOUNT_PATH,
    {
      readOnly: true
    }
  );

  try {
    setStage('sandbox_create_python_context');

    const pythonCtx = await sandbox.createCodeContext({
      language: 'python'
    });

    setStage('workers_ai_initialize');

    const workersai = createWorkersAI({
      binding: env.AI
    });

    setStage('workers_ai_generate_text');

    const result = await generateText({
      model: workersai(MODEL),
      messages: [
        {
          role: 'system',
          content: [
            'Sos un analista de datos.',
            `El CSV privado está disponible en: ${csvPath}`,
            'Usá execute_python las veces estrictamente necesarias para cumplir el prompt.',
            'No inventes datos: analizá exclusivamente el CSV indicado.',
            'El código Python debe leer el CSV desde la ruta indicada.',
            'Después devolvé únicamente la respuesta en el formato exigido por el prompt recibido.'
          ].join(' ')
        },
        {
          role: 'user',
          content: prompt
        }
      ],
      tools: {
        execute_python: tool({
          description:
            'Ejecuta código Python dentro del sandbox para analizar el CSV privado.',
          inputSchema: z.object({
            code: z.string().describe(
              `Código Python a ejecutar. El CSV está disponible en ${csvPath}.`
            )
          }),
          execute: async ({ code }) => {
            setStage('sandbox_run_python');

            const result = await sandbox.runCode(code, {
              context: pythonCtx
            });

            setStage('sandbox_python_completed');

            return formatPythonResult(result);
          }
        })
      },
      maxOutputTokens: 8192,
      stopWhen: () => false
    });

    setStage('workers_ai_response_received');

    const finalText = result.text?.trim();

    if (!finalText) {
      throw new Error(
        `Workers AI no generó respuesta final. finishReason=${result.finishReason} steps=${result.steps.length} toolCalls=${result.toolCalls.length} toolResults=${result.toolResults.length}`
      );
    }

    return finalText;
  } finally {
    setStage('sandbox_unmount_r2');

    try {
      await sandbox.unmountBucket(R2_MOUNT_PATH);
      setStage('sandbox_r2_unmounted');
    } catch (error) {
      console.error({
        event: 'cloudflare_ia_datos_unmount_error',
        error_name: error instanceof Error ? error.name : 'UnknownError',
        error_message:
          error instanceof Error
            ? error.message
            : 'No se pudo desmontar R2'
      });
    }
  }
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

    const authorization = request.headers.get('Authorization');

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

    let stage = 'request_received';

    const setStage: SetStage = (nextStage) => {
      stage = nextStage;

      console.log({
        event: 'cloudflare_ia_datos_stage',
        stage
      });
    };

    try {
      setStage('request_parse_body');

      const body = await request.json<{
        objectKey?: string;
        prompt?: string;
      }>();

      const { objectKey, prompt } = body;

      if (!objectKey || !prompt) {
        return Response.json(
          {
            error: 'Missing objectKey or prompt field'
          },
          {
            status: 400
          }
        );
      }

      if (
        objectKey.startsWith('/') ||
        objectKey.includes('..')
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

      setStage('request_validated');

      const output = await handleAIRequest(
        prompt,
        objectKey,
        env,
        setStage
      );

      setStage('response_ready');

      return Response.json({
        output
      });
    } catch (error) {
      const errorName =
        error instanceof Error
          ? error.name
          : 'UnknownError';

      const errorMessage =
        error instanceof Error
          ? error.message
          : 'Internal Server Error';

      console.error({
        event: 'cloudflare_ia_datos_error',
        stage,
        error_name: errorName,
        error_message: errorMessage
      });

      return Response.json(
        {
          error: errorMessage
        },
        {
          status: 500
        }
      );
    }
  }
} satisfies ExportedHandler<Env>;
