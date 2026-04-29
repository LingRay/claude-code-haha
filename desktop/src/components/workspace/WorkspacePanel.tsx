import { useEffect, useMemo, useState } from 'react'
import { Highlight, type PrismTheme } from 'prism-react-renderer'
import type {
  WorkspaceChangedFile,
  WorkspaceFileStatus,
  WorkspaceTreeEntry,
  WorkspaceTreeResult,
} from '../../api/sessions'
import { useTranslation } from '../../i18n'
import { useShallow } from 'zustand/react/shallow'
import {
  useWorkspacePanelStore,
  type WorkspacePreviewKind,
  type WorkspacePreviewTab,
} from '../../stores/workspacePanelStore'
import { MarkdownRenderer } from '../markdown/MarkdownRenderer'

type WorkspacePanelProps = {
  sessionId: string
}

type TreeNodeProps = {
  sessionId: string
  entry: WorkspaceTreeEntry
  depth: number
  expandedPaths: Set<string>
  treeByPath: Record<string, WorkspaceTreeResult | undefined>
  treeLoadingByPath: Record<string, boolean | undefined>
  treeErrorsByPath: Record<string, string | null | undefined>
  filterQuery: string
  onToggle: (path: string) => void
  onOpenFile: (path: string) => void
  activePath: string | null
}

const FILE_STATUS_META: Record<WorkspaceFileStatus, { label: string; className: string }> = {
  modified: {
    label: 'M',
    className: 'border-[var(--color-warning)]/35 bg-[var(--color-warning)]/12 text-[var(--color-warning)]',
  },
  added: {
    label: 'A',
    className: 'border-[var(--color-success)]/35 bg-[var(--color-success)]/12 text-[var(--color-success)]',
  },
  deleted: {
    label: 'D',
    className: 'border-[var(--color-error)]/35 bg-[var(--color-error)]/12 text-[var(--color-error)]',
  },
  renamed: {
    label: 'R',
    className: 'border-[var(--color-brand)]/35 bg-[var(--color-brand)]/12 text-[var(--color-brand)]',
  },
  untracked: {
    label: 'U',
    className: 'border-[var(--color-tertiary)]/35 bg-[var(--color-tertiary)]/12 text-[var(--color-tertiary)]',
  },
  copied: {
    label: 'C',
    className: 'border-[var(--color-secondary)]/35 bg-[var(--color-secondary)]/12 text-[var(--color-secondary)]',
  },
  type_changed: {
    label: 'T',
    className: 'border-[var(--color-outline)]/45 bg-[var(--color-outline)]/10 text-[var(--color-text-secondary)]',
  },
  unknown: {
    label: '?',
    className: 'border-[var(--color-outline)]/45 bg-[var(--color-outline)]/10 text-[var(--color-text-secondary)]',
  },
}

const EMPTY_TREE_BY_PATH: Record<string, WorkspaceTreeResult | undefined> = {}
const EMPTY_PREVIEW_TABS: WorkspacePreviewTab[] = []
const EMPTY_EXPANDED_PATHS: string[] = []
const PREVIEW_LINE_LIMIT = 420

const workspacePrismTheme: PrismTheme = {
  plain: {
    color: 'var(--color-code-fg)',
    backgroundColor: 'transparent',
  },
  styles: [
    { types: ['comment', 'prolog', 'doctype', 'cdata'], style: { color: 'var(--color-code-comment)', fontStyle: 'italic' } },
    { types: ['string', 'attr-value', 'template-string'], style: { color: 'var(--color-code-string)' } },
    { types: ['keyword', 'selector', 'important', 'atrule'], style: { color: 'var(--color-code-keyword)' } },
    { types: ['function'], style: { color: 'var(--color-code-function)' } },
    { types: ['tag'], style: { color: 'var(--color-code-keyword)' } },
    { types: ['number', 'boolean'], style: { color: 'var(--color-code-number)' } },
    { types: ['operator'], style: { color: 'var(--color-code-fg)' } },
    { types: ['punctuation'], style: { color: 'var(--color-code-punctuation)' } },
    { types: ['variable', 'parameter'], style: { color: 'var(--color-code-fg)' } },
    { types: ['property', 'attr-name'], style: { color: 'var(--color-code-property)' } },
    { types: ['builtin', 'class-name', 'constant', 'symbol'], style: { color: 'var(--color-code-type)' } },
    { types: ['inserted'], style: { color: 'var(--color-code-inserted)' } },
    { types: ['deleted'], style: { color: 'var(--color-code-deleted)' } },
  ],
}

const FILE_BADGE_META: Record<string, { label: string; className: string }> = {
  ts: { label: 'TS', className: 'bg-[#dff0ff] text-[#2b86c5]' },
  tsx: { label: 'TSX', className: 'bg-[#dff0ff] text-[#2b86c5]' },
  js: { label: 'JS', className: 'bg-[#fff3bf] text-[#8a6500]' },
  jsx: { label: 'JSX', className: 'bg-[#fff3bf] text-[#8a6500]' },
  json: { label: '{}', className: 'bg-[#eee9ff] text-[#6f4fb8]' },
  md: { label: 'MD', className: 'bg-[#e9eef3] text-[#5e6872]' },
  css: { label: 'CSS', className: 'bg-[#e4f2ff] text-[#246da6]' },
  html: { label: 'H', className: 'bg-[#ffe7dc] text-[#b9552d]' },
  png: { label: 'IMG', className: 'bg-[#e4f7ed] text-[#287747]' },
  jpg: { label: 'IMG', className: 'bg-[#e4f7ed] text-[#287747]' },
  jpeg: { label: 'IMG', className: 'bg-[#e4f7ed] text-[#287747]' },
  gif: { label: 'IMG', className: 'bg-[#e4f7ed] text-[#287747]' },
  svg: { label: 'SVG', className: 'bg-[#e4f7ed] text-[#287747]' },
}

function makeTreeStateKey(sessionId: string, path: string) {
  return `${sessionId}::${path}`
}

function makePreviewStateKey(sessionId: string, tabId: string) {
  return `${sessionId}::${tabId}`
}

function getSessionScopedRecord<T>(
  record: Record<string, T>,
  sessionId: string,
) {
  const prefix = `${sessionId}::`
  return Object.fromEntries(
    Object.entries(record).filter(([key]) => key.startsWith(prefix)),
  ) as Record<string, T>
}

function getPreviewKindLabel(
  t: ReturnType<typeof useTranslation>,
  kind: WorkspacePreviewKind,
) {
  return kind === 'diff' ? t('workspace.previewKind.diff') : t('workspace.previewKind.file')
}

function getFileExtension(name: string) {
  const cleanName = name.split('/').pop() ?? name
  const lastDot = cleanName.lastIndexOf('.')
  if (lastDot <= 0 || lastDot === cleanName.length - 1) return ''
  return cleanName.slice(lastDot + 1).toLowerCase()
}

function getFileBadgeMeta(name: string) {
  const extension = getFileExtension(name)
  return FILE_BADGE_META[extension] ?? {
    label: extension ? extension.slice(0, 3).toUpperCase() : 'TXT',
    className: 'bg-[#eef0f2] text-[#747b83]',
  }
}

function normalizePrismLanguage(language: string) {
  const lower = language.toLowerCase()
  const map: Record<string, string> = {
    text: 'text',
    typescript: 'typescript',
    ts: 'typescript',
    tsx: 'tsx',
    javascript: 'javascript',
    js: 'javascript',
    jsx: 'jsx',
    markdown: 'markdown',
    md: 'markdown',
    html: 'markup',
    xml: 'markup',
    shell: 'bash',
    sh: 'bash',
    zsh: 'bash',
    diff: 'diff',
  }
  return map[lower] ?? lower
}

function getLanguageFromPath(path: string) {
  return normalizePrismLanguage(getFileExtension(path) || 'text')
}

function isMarkdownPreview(tab: WorkspacePreviewTab) {
  if (tab.kind !== 'file') return false
  const language = (tab.language ?? '').toLowerCase()
  const extension = getFileExtension(tab.path)
  return language === 'markdown' || language === 'md' || extension === 'md' || extension === 'markdown'
}

function FileTypeBadge({ name, subtle = false }: { name: string; subtle?: boolean }) {
  const meta = getFileBadgeMeta(name)
  return (
    <span
      className={`inline-flex h-[18px] min-w-[18px] shrink-0 items-center justify-center rounded-[5px] px-1 font-[var(--font-label)] text-[9px] font-semibold leading-none ${meta.className} ${subtle ? 'opacity-55 grayscale' : ''}`}
      aria-hidden="true"
    >
      {meta.label}
    </span>
  )
}

function InlineHighlightedCode({
  value,
  language,
}: {
  value: string
  language: string
}) {
  return (
    <Highlight
      theme={workspacePrismTheme}
      code={value}
      language={normalizePrismLanguage(language)}
    >
      {({ tokens, getTokenProps }) => (
        <>
          {(tokens[0] ?? []).map((token, tokenIndex) => {
            const { key: tokenKey, ...tokenProps } = getTokenProps({ token, key: tokenIndex })
            return <span key={String(tokenKey)} {...tokenProps} />
          })}
        </>
      )}
    </Highlight>
  )
}

function getInlineStateMessage(
  t: ReturnType<typeof useTranslation>,
  state: WorkspacePreviewTab['state'] | WorkspaceTreeResult['state'] | 'not_git_repo' | undefined,
  fallbackError?: string | null,
) {
  switch (state) {
    case 'loading':
      return t('workspace.previewState.loading')
    case 'binary':
      return t('workspace.previewState.binary')
    case 'too_large':
      return t('workspace.previewState.tooLarge')
    case 'missing':
      return t('workspace.previewState.missing')
    case 'not_git_repo':
      return t('workspace.notGitRepo')
    case 'error':
      return fallbackError || t('workspace.loadError')
    default:
      return fallbackError || t('workspace.loadError')
  }
}

function normalizeFilterQuery(query: string) {
  return query.trim().toLowerCase()
}

function changedFileMatchesFilter(file: WorkspaceChangedFile, query: string) {
  if (!query) return true
  return (
    file.path.toLowerCase().includes(query)
    || file.oldPath?.toLowerCase().includes(query)
    || file.status.toLowerCase().includes(query)
  )
}

function treeEntryMatchesFilter(
  entry: WorkspaceTreeEntry,
  query: string,
  treeByPath: Record<string, WorkspaceTreeResult | undefined>,
): boolean {
  if (!query) return true
  if (entry.name.toLowerCase().includes(query) || entry.path.toLowerCase().includes(query)) {
    return true
  }

  if (!entry.isDirectory) return false
  const childTree = treeByPath[entry.path]
  if (childTree?.state !== 'ok') return false
  return childTree.entries.some((child) => treeEntryMatchesFilter(child, query, treeByPath))
}

function PanelMessage({
  icon,
  message,
  tone = 'muted',
  compact = false,
}: {
  icon: string
  message: string
  tone?: 'muted' | 'error'
  compact?: boolean
}) {
  const toneClass =
    tone === 'error'
      ? 'text-[var(--color-error)]'
      : 'text-[var(--color-text-tertiary)]'

  return (
    <div
      className={`flex items-center gap-2 px-4 ${compact ? 'py-2 text-[11px]' : 'py-8 text-xs'} ${toneClass}`}
      role={tone === 'error' ? 'alert' : 'status'}
    >
      <span className={`material-symbols-outlined shrink-0 text-[16px] ${icon === 'progress_activity' ? 'animate-spin' : ''}`}>
        {icon}
      </span>
      <span className="min-w-0 leading-relaxed">{message}</span>
    </div>
  )
}

function ToolbarIconButton({
  icon,
  label,
  onClick,
}: {
  icon: string
  label: string
  onClick: () => void
}) {
  return (
    <button
      type="button"
      aria-label={label}
      onClick={onClick}
      className="inline-flex h-8 w-8 items-center justify-center rounded-[9px] text-[#777c83] transition-colors hover:bg-[#f1f1f1] hover:text-[#272a2e] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#0a96ff]/45"
    >
      <span className="material-symbols-outlined text-[18px]">{icon}</span>
    </button>
  )
}

function WorkspaceFilterInput({
  value,
  onChange,
}: {
  value: string
  onChange: (value: string) => void
}) {
  const t = useTranslation()

  return (
    <div className="shrink-0 border-b border-[#ececec] px-3 py-2">
      <label className="flex h-8 items-center gap-2 rounded-[9px] border border-[#e5e5e5] bg-white px-2.5 text-[#8b9096] transition-colors focus-within:border-[#0a96ff] focus-within:ring-2 focus-within:ring-[#0a96ff]/10">
        <span className="material-symbols-outlined shrink-0 text-[17px]">search</span>
        <input
          value={value}
          onChange={(event) => onChange(event.target.value)}
          aria-label={t('workspace.filterPlaceholder')}
          placeholder={t('workspace.filterPlaceholder')}
          className="min-w-0 flex-1 bg-transparent text-[13px] text-[#26292d] outline-none placeholder:text-[#a0a4aa]"
        />
        {value.length > 0 && (
          <button
            type="button"
            aria-label={t('workspace.clearFilter')}
            onClick={() => onChange('')}
            className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-[5px] text-[#9ba0a6] transition-colors hover:bg-[#eeeeee] hover:text-[#30343a]"
          >
            <span className="material-symbols-outlined text-[13px]">close</span>
          </button>
        )}
      </label>
    </div>
  )
}

function FileStatusBadge({ status }: { status: WorkspaceFileStatus }) {
  const meta = FILE_STATUS_META[status]
  return (
    <span
      className={`inline-flex h-5 w-5 shrink-0 items-center justify-center rounded border text-[10px] font-semibold ${meta.className}`}
      aria-label={status}
    >
      {meta.label}
    </span>
  )
}

function CodeSurface({ value, language }: { value: string; language: string }) {
  const t = useTranslation()
  const lines = value.split('\n')
  const visibleLines = lines.slice(0, PREVIEW_LINE_LIMIT)
  const visibleCode = visibleLines.join('\n')
  const hiddenLineCount = Math.max(0, lines.length - visibleLines.length)

  return (
    <div className="min-h-0 flex-1 overflow-auto bg-[#fdfdfc]">
      <div className="relative min-w-max py-2">
        <Highlight
          theme={workspacePrismTheme}
          code={visibleCode}
          language={normalizePrismLanguage(language)}
        >
          {({ tokens, getLineProps, getTokenProps }) => (
            <pre
              data-workspace-code=""
              data-testid="workspace-code"
              className="m-0 font-[var(--font-mono)] text-[12px] leading-[1.55]"
              style={{ color: 'var(--color-code-fg)', background: 'transparent' }}
            >
              {tokens.map((line, index) => {
                const { key: lineKey, ...lineProps } = getLineProps({ line, key: index })
                return (
                  <div
                    key={String(lineKey)}
                    {...lineProps}
                    className="grid grid-cols-[48px_minmax(0,1fr)] gap-3 px-3 hover:bg-[#f4f4f3]"
                  >
                    <span className="select-none text-right text-[11px] text-[#96999d]">
                      {index + 1}
                    </span>
                    <span className="whitespace-pre pr-6">
                      {line.length === 1 && line[0]?.empty ? ' ' : line.map((token, tokenIndex) => {
                        const { key: tokenKey, ...tokenProps } = getTokenProps({ token, key: tokenIndex })
                        return <span key={String(tokenKey)} {...tokenProps} />
                      })}
                    </span>
                  </div>
                )
              })}
            </pre>
          )}
        </Highlight>
        {hiddenLineCount > 0 && (
          <div className="sticky bottom-0 border-t border-[#e7e7e7] bg-white/95 px-3 py-2 text-xs text-[var(--color-text-tertiary)] backdrop-blur">
            {t('workspace.previewLineLimit', { count: PREVIEW_LINE_LIMIT })}
          </div>
        )}
      </div>
    </div>
  )
}

function MarkdownSurface({ value }: { value: string }) {
  return (
    <div className="min-h-0 flex-1 overflow-auto bg-[#fdfdfc]">
      <div className="mx-auto w-full max-w-[860px] px-6 py-5">
        <MarkdownRenderer
          content={value}
          variant="document"
          className="workspace-markdown-preview prose-p:text-[14px] prose-p:leading-7 prose-h1:text-[24px] prose-h2:text-[18px] prose-h3:text-[15px] prose-code:text-[12px] prose-pre:my-4"
        />
      </div>
    </div>
  )
}

function DiffSurface({ value, path }: { value: string; path: string }) {
  const t = useTranslation()
  const lines = value.split('\n')
  const visibleLines = lines.slice(0, PREVIEW_LINE_LIMIT)
  const hiddenLineCount = Math.max(0, lines.length - visibleLines.length)
  const language = getLanguageFromPath(path)

  return (
    <div className="min-h-0 flex-1 overflow-auto bg-[#fdfdfc]">
      <div className="relative min-w-max py-2">
        <pre
          data-workspace-code=""
          data-testid="workspace-code"
          className="m-0 font-[var(--font-mono)] text-[12px] leading-[1.55] text-[var(--color-code-fg)]"
        >
          {visibleLines.map((line, index) => {
            const isFileHeader = line.startsWith('diff --') || line.startsWith('--- ') || line.startsWith('+++ ')
            const isHunk = line.startsWith('@@')
            const isAdded = line.startsWith('+') && !line.startsWith('+++')
            const isRemoved = line.startsWith('-') && !line.startsWith('---')
            const isCodeLine = isAdded || isRemoved || line.startsWith(' ')
            const code = isCodeLine ? line.slice(1) : line
            const prefix = isCodeLine ? line[0] : ' '

            return (
              <div
                key={`${index}:${line}`}
                className={`grid grid-cols-[48px_18px_minmax(0,1fr)] gap-2 px-3 ${
                  isAdded
                    ? 'bg-[var(--color-diff-added-bg)]'
                    : isRemoved
                      ? 'bg-[var(--color-diff-removed-bg)]'
                      : isHunk
                        ? 'bg-[var(--color-diff-highlight-bg)]'
                        : 'hover:bg-[#f4f4f3]'
                }`}
              >
                <span className="select-none text-right text-[11px] text-[#96999d]">
                  {index + 1}
                </span>
                <span
                  className={`select-none text-center ${
                    isAdded
                      ? 'text-[var(--color-diff-added-text)]'
                      : isRemoved
                        ? 'text-[var(--color-diff-removed-text)]'
                        : 'text-[#9aa0a6]'
                  }`}
                >
                  {prefix}
                </span>
                <span
                  className={`whitespace-pre pr-6 ${
                    isFileHeader
                      ? 'font-semibold text-[#747a82]'
                      : isHunk
                        ? 'font-semibold text-[#8a6f00]'
                        : ''
                  }`}
                >
                  {isCodeLine ? (
                    code ? <InlineHighlightedCode value={code} language={language} /> : ' '
                  ) : (
                    code || ' '
                  )}
                </span>
              </div>
            )
          })}
        </pre>
        {hiddenLineCount > 0 && (
          <div className="sticky bottom-0 border-t border-[#e7e7e7] bg-white/95 px-3 py-2 text-xs text-[var(--color-text-tertiary)] backdrop-blur">
            {t('workspace.previewLineLimit', { count: PREVIEW_LINE_LIMIT })}
          </div>
        )}
      </div>
    </div>
  )
}

function ImagePreview({ tab }: { tab: WorkspacePreviewTab }) {
  const t = useTranslation()

  if (!tab.dataUrl) {
    return (
      <PanelMessage
        icon="image_not_supported"
        message={tab.error || t('workspace.imagePreviewUnavailable')}
      />
    )
  }

  return (
    <div className="min-h-0 flex-1 overflow-auto bg-[#fdfdfc] p-4">
      <div className="flex min-h-full items-center justify-center">
        <img
          src={tab.dataUrl}
          alt={tab.path}
          className="max-h-full max-w-full rounded-[8px] border border-[#e5e5e5] bg-white object-contain shadow-sm"
        />
      </div>
    </div>
  )
}

function ChangedFileRow({
  file,
  onClick,
}: {
  file: WorkspaceChangedFile
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="mx-2 flex w-[calc(100%-16px)] items-center gap-3 rounded-[7px] px-2 py-2 text-left transition-colors hover:bg-[#f3f3f3]"
    >
      <FileStatusBadge status={file.status} />
      <div className="min-w-0 flex-1">
        <div className="truncate text-[13px] font-medium text-[#282a2d]">{file.path}</div>
        {file.oldPath && (
          <div className="truncate text-[11px] text-[#8b9096]">
            {file.oldPath}
          </div>
        )}
      </div>
      <div className="shrink-0 text-right font-[var(--font-mono)] text-[11px] leading-4">
        <div className="text-[var(--color-success)]">+{file.additions}</div>
        <div className="text-[var(--color-error)]">-{file.deletions}</div>
      </div>
    </button>
  )
}

function TreeNode({
  sessionId,
  entry,
  depth,
  expandedPaths,
  treeByPath,
  treeLoadingByPath,
  treeErrorsByPath,
  filterQuery,
  onToggle,
  onOpenFile,
  activePath,
}: TreeNodeProps) {
  const t = useTranslation()
  const childTree = treeByPath[entry.path]
  const childLoading = treeLoadingByPath[makeTreeStateKey(sessionId, entry.path)] ?? false
  const childError = treeErrorsByPath[makeTreeStateKey(sessionId, entry.path)] ?? null
  const isExpanded = expandedPaths.has(entry.path)
  const isVisuallyExpanded = isExpanded || filterQuery.length > 0
  const indent = 14 + depth * 20

  if (!entry.isDirectory) {
    const isActive = entry.path === activePath
    return (
      <button
        type="button"
        onClick={() => onOpenFile(entry.path)}
        className={`group mx-2 flex h-8 w-[calc(100%-16px)] items-center gap-2 rounded-[7px] pr-2 text-left transition-colors ${
          isActive
            ? 'bg-[#f7f7f7] shadow-[inset_0_0_0_1.5px_#0a96ff]'
            : 'hover:bg-[#f3f3f3]'
        }`}
        style={{ paddingLeft: indent }}
      >
        <FileTypeBadge name={entry.name} subtle={!isActive} />
        <span className="min-w-0 truncate text-[14px] font-medium text-[#282a2d]">{entry.name}</span>
      </button>
    )
  }

  return (
    <div>
      <button
        type="button"
        onClick={() => onToggle(entry.path)}
        aria-expanded={isVisuallyExpanded}
        className="group mx-2 flex h-8 w-[calc(100%-16px)] items-center gap-2 rounded-[7px] pr-2 text-left transition-colors hover:bg-[#f3f3f3]"
        style={{ paddingLeft: indent }}
      >
        <span className="material-symbols-outlined shrink-0 text-[18px] text-[#858a90] transition-colors group-hover:text-[#34373b]">
          {isVisuallyExpanded ? 'expand_more' : 'chevron_right'}
        </span>
        <span className="min-w-0 truncate text-[15px] font-medium text-[#24272b]">{entry.name}</span>
      </button>

      {isVisuallyExpanded && (
        <div className="relative">
          {depth < 4 && (
            <span
              aria-hidden="true"
              className="pointer-events-none absolute bottom-1 top-1 w-px bg-[#e5e5e5]"
              style={{ left: 28 + depth * 20 }}
            />
          )}
          {childLoading && !childTree && (
            <PanelMessage
              compact
              icon="progress_activity"
              message={t('common.loading')}
            />
          )}

          {!childLoading && childError && (
            <PanelMessage compact icon="error" tone="error" message={childError} />
          )}

          {!childLoading && !childError && childTree?.state === 'missing' && (
            <PanelMessage compact icon="folder_off" message={t('workspace.previewState.missing')} />
          )}

          {!childLoading && !childError && childTree?.state === 'error' && (
            <PanelMessage
              compact
              icon="error"
              tone="error"
              message={childTree.error || t('workspace.loadError')}
            />
          )}

          {!childLoading && !childError && childTree?.state === 'ok' && childTree.entries.length === 0 && (
            <PanelMessage compact icon="folder_open" message={t('workspace.noFiles')} />
          )}

          {!childLoading && !childError && childTree?.state === 'ok' && childTree.entries
            .filter((childEntry) => treeEntryMatchesFilter(childEntry, filterQuery, treeByPath))
            .map((childEntry) => (
              <TreeNode
                key={childEntry.path}
                sessionId={sessionId}
                entry={childEntry}
                depth={depth + 1}
                expandedPaths={expandedPaths}
                treeByPath={treeByPath}
                treeLoadingByPath={treeLoadingByPath}
                treeErrorsByPath={treeErrorsByPath}
                filterQuery={filterQuery}
                onToggle={onToggle}
                onOpenFile={onOpenFile}
                activePath={activePath}
              />
            ))}
        </div>
      )}
    </div>
  )
}

export function WorkspacePanel({ sessionId }: WorkspacePanelProps) {
  const t = useTranslation()
  const [filterQuery, setFilterQuery] = useState('')
  const [isViewMenuOpen, setIsViewMenuOpen] = useState(false)
  const width = useWorkspacePanelStore((state) => state.width)
  const isOpen = useWorkspacePanelStore((state) => state.isPanelOpen(sessionId))
  const activeView = useWorkspacePanelStore((state) => state.getActiveView(sessionId))
  const status = useWorkspacePanelStore((state) => state.statusBySession[sessionId])
  const treeByPath = useWorkspacePanelStore((state) => state.treeBySessionPath[sessionId] ?? EMPTY_TREE_BY_PATH)
  const previewTabs = useWorkspacePanelStore((state) => state.previewTabsBySession[sessionId] ?? EMPTY_PREVIEW_TABS)
  const activePreviewTabId = useWorkspacePanelStore((state) => state.activePreviewTabIdBySession[sessionId] ?? null)
  const expandedPaths = useWorkspacePanelStore((state) => state.expandedPathsBySession[sessionId] ?? EMPTY_EXPANDED_PATHS)
  const statusLoading = useWorkspacePanelStore((state) => state.loading.statusBySession[sessionId] ?? false)
  const treeLoadingByPath = useWorkspacePanelStore(
    useShallow((state) => getSessionScopedRecord(state.loading.treeBySessionPath, sessionId)),
  )
  const statusError = useWorkspacePanelStore((state) => state.errors.statusBySession[sessionId] ?? null)
  const treeErrorsByPath = useWorkspacePanelStore(
    useShallow((state) => getSessionScopedRecord(state.errors.treeBySessionPath, sessionId)),
  )
  const setActiveView = useWorkspacePanelStore((state) => state.setActiveView)
  const loadStatus = useWorkspacePanelStore((state) => state.loadStatus)
  const loadTree = useWorkspacePanelStore((state) => state.loadTree)
  const toggleTreeNode = useWorkspacePanelStore((state) => state.toggleTreeNode)
  const openPreview = useWorkspacePanelStore((state) => state.openPreview)
  const closePreview = useWorkspacePanelStore((state) => state.closePreview)
  const closePanel = useWorkspacePanelStore((state) => state.closePanel)

  const rootTree = treeByPath['']
  const rootTreeKey = makeTreeStateKey(sessionId, '')
  const rootTreeLoading = treeLoadingByPath[rootTreeKey] ?? false
  const rootTreeError = treeErrorsByPath[rootTreeKey] ?? null
  const normalizedFilterQuery = normalizeFilterQuery(filterQuery)
  const expandedPathSet = new Set(expandedPaths)
  const activePreviewTab =
    previewTabs.find((tab) => tab.id === activePreviewTabId) ?? previewTabs[previewTabs.length - 1] ?? null
  const activeTreePath = activePreviewTab?.kind === 'file' ? activePreviewTab.path : null
  const filteredChangedFiles = useMemo(
    () => (status?.changedFiles ?? []).filter((file) => changedFileMatchesFilter(file, normalizedFilterQuery)),
    [normalizedFilterQuery, status?.changedFiles],
  )
  const filteredRootEntries = useMemo(
    () => rootTree?.state === 'ok'
      ? rootTree.entries.filter((entry) => treeEntryMatchesFilter(entry, normalizedFilterQuery, treeByPath))
      : [],
    [normalizedFilterQuery, rootTree, treeByPath],
  )
  const activePreviewRequestKey = activePreviewTab
    ? makePreviewStateKey(sessionId, activePreviewTab.id)
    : null
  const activePreviewLoading = useWorkspacePanelStore((state) =>
    activePreviewRequestKey ? state.loading.previewByTabId[activePreviewRequestKey] ?? false : false,
  )
  const activePreviewError = useWorkspacePanelStore((state) =>
    activePreviewRequestKey ? state.errors.previewByTabId[activePreviewRequestKey] ?? null : null,
  )

  useEffect(() => {
    if (!isOpen || activeView !== 'changed' || status || statusLoading || statusError) return
    void loadStatus(sessionId)
  }, [activeView, isOpen, loadStatus, sessionId, status, statusError, statusLoading])

  useEffect(() => {
    if (!isOpen || activeView !== 'all' || rootTree || rootTreeLoading || rootTreeError) return
    void loadTree(sessionId, '')
  }, [activeView, isOpen, loadTree, rootTree, rootTreeError, rootTreeLoading, sessionId])

  if (!isOpen) return null

  const hasPreviewTabs = previewTabs.length > 0
  const panelWidth = hasPreviewTabs ? width : Math.min(width, 520)
  const panelMaxWidth = hasPreviewTabs ? 'min(62%, calc(100% - 328px))' : '36%'
  const panelMinWidth = hasPreviewTabs ? 'min(420px, 54%)' : 'min(340px, 40%)'

  const handleRefresh = () => {
    void loadStatus(sessionId)
    if (activeView === 'all') {
      void loadTree(sessionId, '')
    }
  }

  const handleOpenDiff = (path: string) => {
    void openPreview(sessionId, path, 'diff')
  }

  const handleOpenFile = (path: string) => {
    void openPreview(sessionId, path, 'file')
  }

  const handleSetActiveView = (view: 'changed' | 'all') => {
    setActiveView(sessionId, view)
    setIsViewMenuOpen(false)
  }

  const renderChangedView = () => {
    if (statusLoading && !status) {
      return <PanelMessage icon="progress_activity" message={t('common.loading')} />
    }

    if (status?.state === 'missing_workdir') {
      return <PanelMessage icon="folder_off" message={t('workspace.missingWorkdir')} />
    }

    if (status?.state === 'not_git_repo') {
      return <PanelMessage icon="account_tree" message={t('workspace.notGitRepo')} />
    }

    if (statusError || status?.state === 'error') {
      return (
        <PanelMessage
          icon="error"
          tone="error"
          message={statusError || status?.error || t('workspace.loadError')}
        />
      )
    }

    if (!status) {
      return <PanelMessage icon="progress_activity" message={t('common.loading')} />
    }

    if (status.changedFiles.length === 0) {
      return <PanelMessage icon="check_circle" message={t('workspace.noChanges')} />
    }

    if (filteredChangedFiles.length === 0) {
      return <PanelMessage icon="search_off" message={t('workspace.noMatchingFiles')} />
    }

    return (
      <div>
        {filteredChangedFiles.map((file) => (
          <ChangedFileRow
            key={`${file.path}:${file.status}:${file.oldPath ?? ''}`}
            file={file}
            onClick={() => handleOpenDiff(file.path)}
          />
        ))}
      </div>
    )
  }

  const renderAllFilesView = () => {
    if (rootTreeLoading && !rootTree) {
      return <PanelMessage icon="progress_activity" message={t('common.loading')} />
    }

    if (rootTreeError) {
      return <PanelMessage icon="error" tone="error" message={rootTreeError} />
    }

    if (rootTree?.state === 'missing') {
      return <PanelMessage icon="folder_off" message={t('workspace.missingWorkdir')} />
    }

    if (rootTree?.state === 'error') {
      return <PanelMessage icon="error" tone="error" message={rootTree.error || t('workspace.loadError')} />
    }

    if (!rootTree) {
      return <PanelMessage icon="progress_activity" message={t('common.loading')} />
    }

    if (rootTree.entries.length === 0) {
      return <PanelMessage icon="folder_open" message={t('workspace.noFiles')} />
    }

    if (filteredRootEntries.length === 0) {
      return <PanelMessage icon="search_off" message={t('workspace.noMatchingFiles')} />
    }

    return (
      <div className="py-1">
        {filteredRootEntries.map((entry) => (
          <TreeNode
            key={entry.path}
            sessionId={sessionId}
            entry={entry}
            depth={0}
            expandedPaths={expandedPathSet}
            treeByPath={treeByPath}
            treeLoadingByPath={treeLoadingByPath}
            treeErrorsByPath={treeErrorsByPath}
            filterQuery={normalizedFilterQuery}
            onToggle={(path) => {
              void toggleTreeNode(sessionId, path)
            }}
            onOpenFile={handleOpenFile}
            activePath={activeTreePath}
          />
        ))}
      </div>
    )
  }

  const renderPreviewContent = () => {
    if (!activePreviewTab) {
      return (
        <div className="flex min-h-0 flex-1 items-center justify-center px-4 text-xs text-[var(--color-text-tertiary)]">
          {t('workspace.previewEmpty')}
        </div>
      )
    }

    const kindLabel = getPreviewKindLabel(t, activePreviewTab.kind)
    const state = activePreviewTab.state ?? 'loading'

    return (
      <>
        <div className="flex h-10 shrink-0 items-center gap-1.5 border-b border-[#ededed] bg-white px-3 text-[12px]">
          <span className="truncate text-[#8b8f95]">{status?.repoName || 'workspace'}</span>
          {activePreviewTab.path.split('/').map((segment, index, segments) => (
            <span key={`${segment}:${index}`} className="flex min-w-0 items-center gap-1.5">
              <span className="text-[#b2b5ba]">›</span>
              <span className={`truncate ${index === segments.length - 1 ? 'font-semibold text-[#25272a]' : 'text-[#7b8087]'}`}>
                {segment}
              </span>
            </span>
          ))}
          <span className="ml-auto shrink-0 rounded-[5px] border border-[#ececec] px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-[0.1em] text-[#8a8f96]">
            {kindLabel}
          </span>
        </div>

        {activePreviewLoading || state === 'loading' ? (
          <PanelMessage icon="progress_activity" message={t('workspace.previewState.loading')} />
        ) : state === 'ok' && activePreviewTab.previewType === 'image' ? (
          <ImagePreview tab={activePreviewTab} />
        ) : state === 'ok' && activePreviewTab.kind === 'diff' ? (
          <DiffSurface
            value={activePreviewTab.diff ?? ''}
            path={activePreviewTab.path}
          />
        ) : state === 'ok' && isMarkdownPreview(activePreviewTab) ? (
          <MarkdownSurface value={activePreviewTab.content ?? ''} />
        ) : state === 'ok' ? (
          <CodeSurface
            value={activePreviewTab.content ?? ''}
            language={activePreviewTab.language ?? 'text'}
          />
        ) : (
          <PanelMessage
            icon="error"
            tone={state === 'error' ? 'error' : 'muted'}
            message={getInlineStateMessage(t, state, activePreviewError || activePreviewTab.error || null)}
          />
        )}
      </>
    )
  }

  const renderPreviewTabs = () => (
    <div
      role="tablist"
      aria-label={t('workspace.previewTabs')}
      className="flex h-11 shrink-0 items-center gap-1 overflow-x-auto border-b border-[#ececec] bg-white px-3"
    >
      {previewTabs.length === 0 ? (
        <div className="flex items-center gap-2 px-1.5 text-[12px] text-[#8a8f96]">
          <span className="material-symbols-outlined text-[15px]">docs</span>
          <span>{t('workspace.preview')}</span>
        </div>
      ) : (
        previewTabs.map((tab) => {
          const kindLabel = getPreviewKindLabel(t, tab.kind)
          const isActive = tab.id === activePreviewTab?.id

          return (
            <div
              key={tab.id}
              className={`group flex h-8 min-w-[118px] max-w-[210px] shrink-0 items-center gap-2 rounded-[8px] px-2 text-left text-[13px] transition-colors ${
                isActive
                  ? 'bg-[#f0f0f0] text-[#222529]'
                  : 'text-[#777c83] hover:bg-[#f6f6f6] hover:text-[#2f3338]'
              }`}
            >
              <button
                type="button"
                role="tab"
                aria-selected={isActive}
                onClick={() => {
                  void openPreview(sessionId, tab.path, tab.kind)
                }}
                className="min-w-0 flex flex-1 items-center gap-2 text-left"
              >
                {tab.kind === 'diff' ? (
                  <span className="material-symbols-outlined shrink-0 text-[15px] text-[#8b9096]">difference</span>
                ) : (
                  <FileTypeBadge name={tab.title} subtle={!isActive} />
                )}
                <span className="min-w-0 flex-1 truncate">{tab.title}</span>
              </button>
              <button
                type="button"
                aria-label={`${t('workspace.closeTab')} ${tab.title} ${kindLabel}`}
                onClick={() => {
                  closePreview(sessionId, tab.id)
                }}
                className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-[5px] text-[#9a9fa6] opacity-0 transition-colors hover:bg-[#e4e4e4] hover:text-[#30343a] group-hover:opacity-100 focus-visible:opacity-100"
              >
                <span className="material-symbols-outlined text-[13px] leading-none">close</span>
              </button>
            </div>
          )
        })
      )}
    </div>
  )

  return (
    <aside
      data-testid="workspace-panel"
      className="flex h-full shrink-0 border-l border-[#e7e7e7] bg-white"
      style={{ width: panelWidth, maxWidth: panelMaxWidth, minWidth: panelMinWidth }}
    >
      {hasPreviewTabs && (
        <div className="flex min-w-0 flex-1 flex-col border-r border-[#e7e7e7] bg-white">
          {renderPreviewTabs()}
          {renderPreviewContent()}
        </div>
      )}

      <div
        className={`${hasPreviewTabs ? 'basis-[32%] min-w-[220px] max-w-[320px]' : 'w-full'} flex h-full shrink-0 flex-col bg-white`}
      >
        <div className="flex h-12 shrink-0 items-center gap-2 border-b border-[#ececec] px-3">
          <div className="relative min-w-0">
            <button
              type="button"
              aria-label={activeView === 'changed' ? t('workspace.changedFiles') : t('workspace.allFiles')}
              aria-haspopup="menu"
              aria-expanded={isViewMenuOpen}
              onClick={() => setIsViewMenuOpen((open) => !open)}
              className="inline-flex min-w-0 max-w-full items-center gap-1 rounded-[8px] px-2.5 py-1.5 text-[18px] font-semibold leading-none text-[#202327] transition-colors hover:bg-[#f4f4f4] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#0a96ff]/35"
            >
              <span className="truncate">
                {activeView === 'changed' ? t('workspace.changedFiles') : t('workspace.allFiles')}
              </span>
              <span className="material-symbols-outlined shrink-0 text-[18px] font-normal text-[#858a90]">expand_more</span>
            </button>
            {isViewMenuOpen && (
              <div
                role="menu"
                className="absolute left-0 top-[calc(100%+6px)] z-30 min-w-[132px] overflow-hidden rounded-[10px] border border-[#e6e6e6] bg-white py-1 shadow-[0_10px_28px_rgba(20,20,20,0.12)]"
              >
                {(['changed', 'all'] as const).map((view) => {
                  const selected = activeView === view
                  return (
                    <button
                      key={view}
                      type="button"
                      role="menuitem"
                      onClick={() => handleSetActiveView(view)}
                      className={`flex h-8 w-full items-center gap-2 px-3 text-left text-[13px] transition-colors ${
                        selected ? 'bg-[#f4f4f4] text-[#202327]' : 'text-[#4b5056] hover:bg-[#f7f7f7]'
                      }`}
                    >
                      <span className="min-w-0 flex-1 truncate">
                        {view === 'changed' ? t('workspace.changedFiles') : t('workspace.allFiles')}
                      </span>
                      {selected && (
                        <span className="material-symbols-outlined text-[15px] text-[#0a96ff]">check</span>
                      )}
                    </button>
                  )
                })}
              </div>
            )}
          </div>

          <div className="ml-auto flex shrink-0 items-center gap-1">
            <ToolbarIconButton
              icon="refresh"
              label={t('workspace.refresh')}
              onClick={handleRefresh}
            />
            <ToolbarIconButton
              icon="close"
              label={t('workspace.closePanel')}
              onClick={() => closePanel(sessionId)}
            />
          </div>
        </div>

        <WorkspaceFilterInput value={filterQuery} onChange={setFilterQuery} />

        <div className="min-h-0 flex-1 overflow-auto py-2">
          {activeView === 'changed' ? renderChangedView() : renderAllFilesView()}
        </div>
      </div>
    </aside>
  )
}
