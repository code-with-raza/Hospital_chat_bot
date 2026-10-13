import { Fragment, ReactNode } from 'react'

type MarkdownContentProps = {
  content: string
}

function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const parts: ReactNode[] = []
  const emphasis = /(\*\*|__)(.+?)\1/g
  let cursor = 0
  let match: RegExpExecArray | null

  while ((match = emphasis.exec(text)) !== null) {
    if (match.index > cursor) {
      parts.push(text.slice(cursor, match.index))
    }
    parts.push(<strong key={`${keyPrefix}-strong-${match.index}`}>{match[2]}</strong>)
    cursor = match.index + match[0].length
  }

  if (cursor < text.length) parts.push(text.slice(cursor))
  return parts
}

function listItem(line: string): { kind: 'ul' | 'ol'; text: string } | null {
  const unordered = line.match(/^\s*[-*+]\s+(.+)$/)
  if (unordered) return { kind: 'ul', text: unordered[1] }

  const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/)
  if (ordered) return { kind: 'ol', text: ordered[1] }

  return null
}

export default function MarkdownContent({ content }: MarkdownContentProps) {
  const lines = content.replace(/\r\n?/g, '\n').split('\n')
  const blocks: ReactNode[] = []
  let index = 0

  while (index < lines.length) {
    if (!lines[index].trim()) {
      index += 1
      continue
    }

    const firstItem = listItem(lines[index])
    if (firstItem) {
      const items: ReactNode[] = []
      const kind = firstItem.kind
      while (index < lines.length) {
        const item = listItem(lines[index])
        if (!item || item.kind !== kind) break
        items.push(<li key={`item-${index}`}>{renderInline(item.text, `item-${index}`)}</li>)
        index += 1
      }
      blocks.push(kind === 'ul'
        ? <ul key={`list-${index}`}>{items}</ul>
        : <ol key={`list-${index}`}>{items}</ol>)
      continue
    }

    const paragraph: string[] = []
    while (index < lines.length && lines[index].trim() && !listItem(lines[index])) {
      paragraph.push(lines[index])
      index += 1
    }
    blocks.push(
      <p key={`paragraph-${index}`}>
        {paragraph.map((line, lineIndex) => (
          <Fragment key={`line-${lineIndex}`}>
            {lineIndex > 0 && <br />}
            {renderInline(line, `paragraph-${index}-${lineIndex}`)}
          </Fragment>
        ))}
      </p>,
    )
  }

  return <div className="markdown-content">{blocks}</div>
}
