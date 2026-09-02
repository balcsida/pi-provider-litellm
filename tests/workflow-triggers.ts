export interface WorkflowTriggers {
  branches: string[];
  tags: string[];
}

function unquote(value: string): string {
  const trimmed = value.trim();
  if (
    trimmed.length >= 2 &&
    ((trimmed.startsWith("'") && trimmed.endsWith("'")) || (trimmed.startsWith('"') && trimmed.endsWith('"')))
  ) {
    return trimmed.slice(1, -1);
  }
  return trimmed;
}

function parseInlineList(value: string): string[] {
  const trimmed = value.trim();
  if (!trimmed.startsWith("[") || !trimmed.endsWith("]")) return [];
  return trimmed.slice(1, -1).split(",").map(unquote).filter(Boolean);
}

export function parsePushTriggers(workflow: string): WorkflowTriggers {
  const lines = workflow.split(/\r?\n/);
  const onIndex = lines.findIndex((line) => /^on:\s*$/.test(line));
  if (onIndex === -1) return { branches: [], tags: [] };

  const pushIndex = lines.findIndex((line, index) => index > onIndex && /^ {2}push:\s*$/.test(line));
  if (pushIndex === -1) return { branches: [], tags: [] };

  const result: WorkflowTriggers = { branches: [], tags: [] };
  let section: keyof WorkflowTriggers | undefined;

  for (const line of lines.slice(pushIndex + 1)) {
    if (/^ {2}\S/.test(line)) break;

    const key = /^ {4}(branches|tags):\s*(.*)$/.exec(line);
    if (key) {
      section = key[1] as keyof WorkflowTriggers;
      result[section].push(...parseInlineList(key[2]));
      continue;
    }

    const item = /^ {6}-\s+(.+)$/.exec(line);
    if (section && item) result[section].push(unquote(item[1]));
  }

  return result;
}
