const assert = require('node:assert/strict')
const fs = require('node:fs')
const test = require('node:test')
const React = require('react')
const { renderToStaticMarkup } = require('react-dom/server')
const ts = require('typescript')

for (const extension of ['.tsx', '.ts']) {
  require.extensions[extension] = (loadedModule, filename) => {
    const source = fs.readFileSync(filename, 'utf8')
    const output = ts.transpileModule(source, {
      compilerOptions: {
        module: ts.ModuleKind.CommonJS,
        target: ts.ScriptTarget.ES2020,
        jsx: ts.JsxEmit.ReactJSX,
        esModuleInterop: true,
      },
      fileName: filename,
    }).outputText
    loadedModule._compile(output, filename)
  }
}

const MessageBubble = require('../src/components/MessageBubble.tsx').default
const renderReply = (content, type = 'text', data) => renderToStaticMarkup(
  React.createElement(MessageBubble, {
    message: { role: 'assistant', content, type, data },
    onSelectDoctor: () => {},
  }),
)

test('shared chat renderer formats Markdown from different assistant response types', () => {
  const knowledgeReply = renderReply('**ICU visiting hours:** 5:00 PM–6:00 PM daily.', 'knowledge_answer')
  const doctorReply = renderReply('**Available doctors**\n\n- Dr. Hina Malik\n- Dr. Usman Tariq', 'doctor_list', [])
  const appointmentReply = renderReply('Next steps:\n1. Choose a date.\n2. Choose a time.', 'appointment_confirmation')

  assert.match(knowledgeReply, /<strong>ICU visiting hours:<\/strong>/)
  assert.doesNotMatch(knowledgeReply, /\*\*/)
  assert.match(doctorReply, /<ul><li>Dr\. Hina Malik<\/li><li>Dr\. Usman Tariq<\/li><\/ul>/)
  assert.match(doctorReply, /<strong>Available doctors<\/strong>/)
  assert.match(appointmentReply, /<ol><li>Choose a date\.<\/li><li>Choose a time\.<\/li><\/ol>/)
})

test('renderer preserves plain text and escapes HTML instead of injecting it', () => {
  const markup = renderReply('Plain reply\nsecond line\n\nNew paragraph.\n<script>alert(1)</script>')

  assert.match(markup, /Plain reply<br\s*\/>second line/)
  assert.match(markup, /<p>New paragraph\.<br\s*\/>/)
  assert.match(markup, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/)
  assert.doesNotMatch(markup, /<script>/)
})
