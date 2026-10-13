// Build the Word 2-pager from the Markdown source so both formats stay identical.
//   node docs/build_docx.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, LevelFormat, BorderStyle, Footer, PageNumber, ImageRun,
} = require("docx");

const SRC = path.join(__dirname, "AD_Multimodal_Workflow_2pager.md");
const OUT = path.join(__dirname, "AD_Multimodal_Workflow_2pager.docx");
const FONT = "Calibri";
const SIZE = 17;            // half-points (8.5 pt)
const ACCENT = "1F3A5F";
const PAGE_W = 12240, MARGIN = 720, CONTENT_W = PAGE_W - 2 * MARGIN;

// inline markdown: **bold**, `code`
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, size: SIZE, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, size: SIZE, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), font: "Consolas", size: SIZE - 1, color: "444444", ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, size: SIZE, ...base }));
  return out;
}

function table(rows) {
  const cells = rows.map(r => r.replace(/^\||\|$/g, "").split("|").map(s => s.trim()));
  const header = cells[0], body = cells.slice(2);
  const n = header.length;
  const widths = n === 2 ? [2300, CONTENT_W - 2300] : [Math.round(CONTENT_W * 0.34), ...Array(n - 1).fill(Math.round(CONTENT_W * 0.66 / (n - 1)))];
  widths[n - 1] = CONTENT_W - widths.slice(0, n - 1).reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "BFC6D1" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const mk = (txt, i, isHead, isBold) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA }, borders,
    shading: isHead ? { fill: ACCENT, type: ShadingType.CLEAR, color: "auto" } : (isBold ? { fill: "E8EEF5", type: ShadingType.CLEAR, color: "auto" } : undefined),
    margins: { top: 30, bottom: 30, left: 80, right: 80 },
    children: [new Paragraph({ spacing: { before: 0, after: 0 }, children: runs(txt, isHead ? { bold: true, color: "FFFFFF" } : {}) })],
  });
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => mk(h, i, true, false)) }),
      ...body.map(r => new TableRow({ children: r.map((c, i) => mk(c, i, false, r[0].startsWith("**"))) }))],
  });
}

// two consecutive ![caption](path) lines -> side-by-side figures with captions
function pngSize(buf) { return [buf.readUInt32BE(16), buf.readUInt32BE(20)]; }
function figurePair(items) {
  const colW = Math.floor(CONTENT_W / items.length);
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const cells = items.map(([cap, rel]) => {
    const buf = fs.readFileSync(path.join(__dirname, rel));
    const [w, h] = pngSize(buf);
    const wpx = Math.round(colW / 1440 * 96) - 8;
    return new TableCell({
      width: { size: colW, type: WidthType.DXA }, borders: { top: none, bottom: none, left: none, right: none },
      children: [
        new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ type: "png", data: buf,
          transformation: { width: wpx, height: Math.round(wpx * h / w) }, altText: { title: cap, description: cap, name: cap } })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [new TextRun({ text: cap, italics: true, font: FONT, size: 15, color: "555555" })] }),
      ],
    });
  });
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: items.map(() => colW), rows: [new TableRow({ children: cells })] });
}

const lines = fs.readFileSync(SRC, "utf8").split("\n");
const children = [];
for (let i = 0; i < lines.length; i++) {
  const l = lines[i];
  if (!l.trim()) continue;
  if (l.startsWith("# ")) {
    children.push(new Paragraph({ spacing: { after: 40 }, children: [new TextRun({ text: l.slice(2), bold: true, font: FONT, size: 28, color: ACCENT })] }));
  } else if (l.startsWith("## ")) {
    children.push(new Paragraph({
      spacing: { before: 110, after: 40 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 1 } },
      children: [new TextRun({ text: l.slice(3), bold: true, font: FONT, size: 21, color: ACCENT })],
    }));
  } else if (l.startsWith("![")) {
    const items = [];
    while (i < lines.length && lines[i].startsWith("![")) {
      const m = lines[i].match(/^!\[(.*)\]\((.*)\)/);
      items.push([m[1], m[2]]);
      i++;
    }
    i--;
    children.push(figurePair(items));
  } else if (l.startsWith("|")) {
    const rows = [];
    while (i < lines.length && lines[i].startsWith("|")) rows.push(lines[i++]);
    i--;
    children.push(table(rows));
    children.push(new Paragraph({ spacing: { after: 20 }, children: [] }));
  } else if (l.startsWith("- ")) {
    children.push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 20 }, children: runs(l.slice(2)) }));
  } else if (/^\d+\. /.test(l)) {
    children.push(new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 20 }, children: runs(l.replace(/^\d+\. /, "")) }));
  } else {
    children.push(new Paragraph({ spacing: { after: 50 }, children: runs(l) }));
  }
}

const doc = new Document({
  creator: "YO",
  title: "Multimodal AD Diagnostic Model: Workflow and Pipelines",
  styles: { default: { document: { run: { font: FONT, size: SIZE } } } },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 300, hanging: 200 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 300, hanging: 220 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 15840 }, margin: { top: 640, bottom: 640, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [
      new TextRun({ text: "Synthetic data, research use only  |  Page ", font: FONT, size: 14, color: "888888" }),
      new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 14, color: "888888" })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then(b => { fs.writeFileSync(OUT, b); console.log("wrote", OUT); });
