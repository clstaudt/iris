/** Markdown for assistant replies. Replies can quote web pages the browse
 *  tool fetched, so the HTML is sanitized before it reaches the DOM. */

import DOMPurify from "dompurify";
import { marked } from "marked";

marked.setOptions({ gfm: true, breaks: true });

// Links leave the app: main.ts hands window.open targets to the browser
DOMPurify.addHook("afterSanitizeAttributes", (node) => {
  if (node.tagName === "A") {
    node.setAttribute("target", "_blank");
    node.setAttribute("rel", "noopener noreferrer");
  }
});

export function renderMarkdown(text: string): string {
  const html = marked.parse(text, { async: false }) as string;
  return DOMPurify.sanitize(html, {
    FORBID_TAGS: ["img", "style", "form", "input", "button"],
  });
}
