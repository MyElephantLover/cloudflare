import { createWorkersAI } from "workers-ai-provider";
import { routeAgentRequest, callable, type Schedule } from "agents";
import { getSchedulePrompt, scheduleSchema } from "agents/schedule";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import {
  streamText,
  convertToModelMessages,
  pruneMessages,
  tool,
  stepCountIs,
} from "ai";
import { z } from "zod";
import Firecrawl from "@mendable/firecrawl-js";

interface Env {
  AI: Ai;
  FIRECRAWL_API_KEY: string;
}

const NEWS_SOURCES: Record<"bbc" | "cnn" | "fox", string> = {
  bbc: "https://www.bbc.com/news",
  cnn: "https://www.cnn.com",
  fox: "https://www.foxnews.com",
};

export class ChatAgent extends AIChatAgent<Env> {
  waitForMcpConnections = true;

  onStart() {
    this.mcp.configureOAuthCallback({
      customHandler: (result) => {
        if (result.authSuccess) {
          return new Response("<script>window.close();</script>", {
            headers: { "content-type": "text/html" },
            status: 200,
          });
        }

        return new Response(
          `Authentication Failed: ${result.authError || "Unknown error"}`,
          {
            headers: { "content-type": "text/plain" },
            status: 400,
          }
        );
      },
    });
  }

  @callable()
  async addServer(name: string, url: string, host: string) {
    return await this.addMcpServer(name, url, { callbackHost: host });
  }

  @callable()
  async removeServer(serverId: string) {
    await this.removeMcpServer(serverId);
  }

  async onChatMessage(_onFinish: unknown, options?: OnChatMessageOptions) {
    const mcpTools = this.mcp.getAITools();
    const workersai = createWorkersAI({ binding: this.env.AI });

    const result = streamText({
      model: workersai("@cf/zai-org/glm-4.7-flash"),
      system: `You are a helpful assistant.

You can:
- check weather
- get the user's timezone
- run calculations
- fetch top news from BBC, CNN, and Fox
- schedule tasks

Important:
- Before running getTopNews or scheduleTask, wait for user approval.
- If approval is denied, explain that the action was not performed.

${getSchedulePrompt({ date: new Date() })}`,
      messages: pruneMessages({
        messages: await convertToModelMessages(this.messages),
        toolCalls: "before-last-2-messages",
      }),
      tools: {
        ...mcpTools,

        getWeather: tool({
          description: "Get the current weather for a city",
          inputSchema: z.object({
            city: z.string().describe("City name"),
          }),
          execute: async ({ city }) => {
            const conditions = ["sunny", "cloudy", "rainy", "snowy"];
            const temp = Math.floor(Math.random() * 30) + 5;

            return {
              city,
              temperature: temp,
              condition:
                conditions[Math.floor(Math.random() * conditions.length)],
              unit: "celsius",
            };
          },
        }),

        getTopNews: tool({
          description:
            "Fetch top news from one supported news source: bbc, cnn, or fox.",
          inputSchema: z.object({
            source: z.enum(["bbc", "cnn", "fox"]),
          }),

          // HITL: always require approval before scraping
          needsApproval: true,

          execute: async ({ source }) => {
            const url = NEWS_SOURCES[source];
            const app = new Firecrawl({
              apiKey: this.env.FIRECRAWL_API_KEY,
            });

            try {
              const scrapeResult = await app.scrape(url, {
                formats: ["markdown"],
              });

              const markdown =
                typeof scrapeResult === "object" &&
                scrapeResult !== null &&
                "markdown" in scrapeResult
                  ? String((scrapeResult as { markdown?: unknown }).markdown ?? "")
                  : JSON.stringify(scrapeResult);

              const headlines = markdown
                .split("\n")
                .map((line) => line.trim())
                .filter((line) => line.length > 25)
                .filter((line) => !line.startsWith("!["))
                .slice(0, 10);

              return {
                source,
                url,
                headlines,
              };
            } catch (error) {
              return {
                source,
                url,
                error:
                  error instanceof Error ? error.message : "Unknown Firecrawl error",
              };
            }
          },
        }),

        getUserTimezone: tool({
          description:
            "Get the user's timezone from their browser. Use this when you need to know the user's local time.",
          inputSchema: z.object({}),
        }),

        calculate: tool({
          description:
            "Perform a math calculation with two numbers. Requires user approval for large numbers.",
          inputSchema: z.object({
            a: z.number().describe("First number"),
            b: z.number().describe("Second number"),
            operator: z
              .enum(["+", "-", "*", "/", "%"])
              .describe("Arithmetic operator"),
          }),
          needsApproval: async ({ a, b }) =>
            Math.abs(a) > 1000 || Math.abs(b) > 1000,
          execute: async ({ a, b, operator }) => {
            const ops: Record<string, (x: number, y: number) => number> = {
              "+": (x, y) => x + y,
              "-": (x, y) => x - y,
              "*": (x, y) => x * y,
              "/": (x, y) => x / y,
              "%": (x, y) => x % y,
            };

            if (operator === "/" && b === 0) {
              return { error: "Division by zero" };
            }

            return {
              expression: `${a} ${operator} ${b}`,
              result: ops[operator](a, b),
            };
          },
        }),

        scheduleTask: tool({
          description:
            "Schedule a task to be executed later. Always requires user approval.",
          inputSchema: scheduleSchema,

          // HITL: always require approval before scheduling
          needsApproval: true,

          execute: async ({ when, description }) => {
            if (when.type === "no-schedule") {
              return "Not a valid schedule input";
            }

            const input =
              when.type === "scheduled"
                ? when.date
                : when.type === "delayed"
                ? when.delayInSeconds
                : when.type === "cron"
                ? when.cron
                : null;

            if (!input) return "Invalid schedule type";

            try {
              this.schedule(input, "executeTask", description);
              return `Task scheduled: "${description}" (${when.type}: ${input})`;
            } catch (error) {
              return `Error scheduling task: ${error}`;
            }
          },
        }),

        getScheduledTasks: tool({
          description: "List all tasks that have been scheduled",
          inputSchema: z.object({}),
          execute: async () => {
            const tasks = this.getSchedules();
            return tasks.length > 0 ? tasks : "No scheduled tasks found.";
          },
        }),

        cancelScheduledTask: tool({
          description: "Cancel a scheduled task by its ID",
          inputSchema: z.object({
            taskId: z.string().describe("The ID of the task to cancel"),
          }),

          // optional: also gate cancellation behind approval
          needsApproval: true,

          execute: async ({ taskId }) => {
            try {
              this.cancelSchedule(taskId);
              return `Task ${taskId} cancelled.`;
            } catch (error) {
              return `Error cancelling task: ${error}`;
            }
          },
        }),
      },
      stopWhen: stepCountIs(5),
      abortSignal: options?.abortSignal,
    });

    return result.toUIMessageStreamResponse();
  }

  async executeTask(description: string, _task: Schedule<string>) {
    console.log(`Executing scheduled task: ${description}`);

    this.broadcast(
      JSON.stringify({
        type: "scheduled-task",
        description,
        timestamp: new Date().toISOString(),
      })
    );
  }
}

export default {
  async fetch(request: Request, env: Env) {
    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  },
} satisfies ExportedHandler<Env>;
