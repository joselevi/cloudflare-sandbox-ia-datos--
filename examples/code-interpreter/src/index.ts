import { ContainerProxy, getSandbox } from '@cloudflare/sandbox';
import { generateText, stepCountIs, tool } from 'ai';
import { createWorkersAI } from 'workers-ai-provider';
import { z } from 'zod';

export { ContainerProxy };
export { Sandbox } from '@cloudflare/sandbox';

const API_PATH = '/run';
const MODEL = '@cf/openai/gpt-oss-120b' as const;
const SANDBOX_WORKSPACE = '/workspace';
const HEARTBEAT_INTERVAL_MS = 15000;

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

  // El Worker lee el CSV por binding de R2 y lo escribe en el sandbox con
  // una ruta única por request: sin mounts compartidos ni desmontajes que
  // puedan pisar lecturas de requests solapadas.
  const csvPath = `${SANDBOX_WORKSPACE}/${objectKey.replaceAll('/', '_')}`;

  setStage('worker_read_r2');

  const r2Object = await env.IA_DATOS_BUCKET.get(objectKey);

  if (!r2Object) {
    throw new Error(`Objeto R2 no encontrado: ${objectKey}`);
  }

  const csvText = await r2Object.text();

  setStage('sandbox_write_file');

  await sandbox.writeFile(csvPath, csvText);

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
            'El CSV usa separador ";" y encoding UTF-8.',
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
      stopWhen: stepCountIs(10)
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
    // Limpieza best-effort: el sandbox es compartido entre requests y los
    // CSV acumulados ocupan espacio en su filesystem.
    setStage('sandbox_cleanup_file');

    try {
      await sandbox.deleteFile(csvPath);
      setStage('sandbox_cleanup_done');
    } catch (error) {
      console.error({
        event: 'cloudflare_ia_datos_cleanup_error',
        path: csvPath,
        error_name: error instanceof Error ? error.name : 'UnknownError',
        error_message:
          error instanceof Error
            ? error.message
            : 'No se pudo eliminar el CSV del sandbox'
      });
    }
  }
}

// El análisis puede tardar minutos: la respuesta se transmite con heartbeats
// para que el edge nunca vea la conexión sin bytes y la corte por inactividad.
// Los espacios iniciales son inocuos: JSON.parse los tolera y el texto se trimea.
// El envelope {"output": ...} es el contrato que espera el extractor del lado Node.
function streamingResponse(
  work: (setStage: SetStage) => Promise<string>,
  setStage: SetStage
): Response {
  const encoder = new TextEncoder();
  let heartbeat: ReturnType<typeof setInterval> | undefined;

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        controller.enqueue(encoder.encode(' '));

        heartbeat = setInterval(() => {
          controller.enqueue(encoder.encode(' '));
        }, HEARTBEAT_INTERVAL_MS);

        const output = await work(setStage);

        setStage('response_ready');

        controller.enqueue(encoder.encode(JSON.stringify({ output })));
      } catch (error) {
        const errorMessage =
          error instanceof Error
            ? error.message
            : 'Internal Server Error';

        console.error({
          event: 'cloudflare_ia_datos_error',
          stage: 'streaming_work',
          error_name: error instanceof Error ? error.name : 'UnknownError',
          error_message: errorMessage
        });

        controller.enqueue(encoder.encode(JSON.stringify({ error: errorMessage })));
      } finally {
        if (heartbeat) {
          clearInterval(heartbeat);
        }

        controller.close();
      }
    },
    cancel() {
      if (heartbeat) {
        clearInterval(heartbeat);
      }
    }
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'application/json'
    }
  });
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

      return streamingResponse(
        () => handleAIRequest(prompt, objectKey, env, setStage),
        setStage
      );
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
