// @vitest-environment jsdom

// @ts-expect-error jsdom is installed in this workspace without local type declarations
import { JSDOM } from 'jsdom'
import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

if (typeof document === 'undefined') {
  const dom = new JSDOM('<!doctype html><html><body></body></html>', {
    url: 'http://localhost/',
  })
  const { window } = dom

  Object.assign(globalThis, {
    window,
    document: window.document,
    navigator: window.navigator,
    localStorage: window.localStorage,
    HTMLElement: window.HTMLElement,
    HTMLButtonElement: window.HTMLButtonElement,
    MutationObserver: window.MutationObserver,
    Node: window.Node,
    Event: window.Event,
    MouseEvent: window.MouseEvent,
    KeyboardEvent: window.KeyboardEvent,
    getComputedStyle: window.getComputedStyle.bind(window),
    IS_REACT_ACT_ENVIRONMENT: true,
  })
}

type WorkspaceApiMocks = {
  getWorkspaceStatusMock: ReturnType<typeof vi.fn>
  getWorkspaceTreeMock: ReturnType<typeof vi.fn>
  getWorkspaceFileMock: ReturnType<typeof vi.fn>
  getWorkspaceDiffMock: ReturnType<typeof vi.fn>
}

var mocks: WorkspaceApiMocks | undefined

function getMocks() {
  if (!mocks) {
    throw new Error('Workspace API mocks were not initialized')
  }
  return mocks
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

async function setWorkspaceState(
  updater:
    | ReturnType<typeof useWorkspacePanelStore.getInitialState>
    | Parameters<typeof useWorkspacePanelStore.setState>[0],
) {
  await act(() => {
    useWorkspacePanelStore.setState(updater as Parameters<typeof useWorkspacePanelStore.setState>[0], true)
  })
}

async function setSettingsState(
  updater:
    | ReturnType<typeof useSettingsStore.getInitialState>
    | Parameters<typeof useSettingsStore.setState>[0],
) {
  await act(() => {
    useSettingsStore.setState(updater as Parameters<typeof useSettingsStore.setState>[0], true)
  })
}

async function renderPanel(sessionId: string) {
  let view!: ReturnType<typeof render>
  await act(() => {
    view = render(<WorkspacePanel sessionId={sessionId} />)
  })
  return view
}

async function clickElement(element: Element) {
  await act(() => {
    fireEvent.click(element)
  })
}

vi.mock('../../api/sessions', () => ({
  sessionsApi: (() => {
    if (!mocks) {
      mocks = {
        getWorkspaceStatusMock: vi.fn(),
        getWorkspaceTreeMock: vi.fn(),
        getWorkspaceFileMock: vi.fn(),
        getWorkspaceDiffMock: vi.fn(),
      }
    }

    return {
      getWorkspaceStatus: mocks.getWorkspaceStatusMock,
      getWorkspaceTree: mocks.getWorkspaceTreeMock,
      getWorkspaceFile: mocks.getWorkspaceFileMock,
      getWorkspaceDiff: mocks.getWorkspaceDiffMock,
    }
  })(),
}))

import { useSettingsStore } from '../../stores/settingsStore'
import { useWorkspacePanelStore } from '../../stores/workspacePanelStore'
import { WorkspacePanel } from './WorkspacePanel'

describe('WorkspacePanel', () => {
  const workspaceInitialState = useWorkspacePanelStore.getInitialState()
  const settingsInitialState = useSettingsStore.getInitialState()

  beforeEach(async () => {
    vi.clearAllMocks()
    await setWorkspaceState(workspaceInitialState)
    await setSettingsState({ ...settingsInitialState, locale: 'en' })
  })

  afterEach(async () => {
    cleanup()
    await setWorkspaceState(workspaceInitialState)
    await setSettingsState(settingsInitialState)
    vi.restoreAllMocks()
  })

  it('stays hidden when the panel is closed', async () => {
    const view = await renderPanel('session-hidden')

    expect(view.queryByTestId('workspace-panel')).toBeNull()
  })

  it('loads changed status on open and opens a diff preview from the changed view', async () => {
    const statusRequest = deferred<{
      state: 'ok'
      workDir: string
      repoName: string
      branch: string
      isGitRepo: true
      changedFiles: Array<{
        path: string
        status: 'modified'
        additions: number
        deletions: number
      }>
    }>()
    const diffRequest = deferred<{
      state: 'ok'
      path: string
      diff: string
    }>()

    getMocks().getWorkspaceStatusMock.mockReturnValue(statusRequest.promise)
    getMocks().getWorkspaceDiffMock.mockReturnValue(diffRequest.promise)

    await act(() => {
      useWorkspacePanelStore.getState().openPanel('session-changed')
    })

    const view = await renderPanel('session-changed')

    const compactPanel = view.getByTestId('workspace-panel')
    expect(compactPanel.style.width).toBe('520px')
    expect(compactPanel.style.maxWidth).toBe('36%')
    expect(compactPanel.style.minWidth).toBe('min(340px, 40%)')

    await waitFor(() => {
      expect(getMocks().getWorkspaceStatusMock).toHaveBeenCalledWith('session-changed')
    })

    await act(async () => {
      statusRequest.resolve({
        state: 'ok',
        workDir: '/repo',
        repoName: 'repo',
        branch: 'main',
        isGitRepo: true,
        changedFiles: [
          {
            path: 'src/app.ts',
            status: 'modified',
            additions: 4,
            deletions: 1,
          },
        ],
      })
      await statusRequest.promise
    })

    expect(view.getByPlaceholderText('Filter files...')).toBeTruthy()

    await clickElement(await view.findByText('src/app.ts'))

    await waitFor(() => {
      expect(getMocks().getWorkspaceDiffMock).toHaveBeenCalledWith('session-changed', 'src/app.ts')
    })

    await act(async () => {
      diffRequest.resolve({
        state: 'ok',
        path: 'src/app.ts',
        diff: '@@ -1 +1 @@\n-console.log("old")\n+console.log("new")',
      })
      await diffRequest.promise
    })

    await waitFor(() => {
      expect(view.getByTestId('workspace-code').textContent).toContain('console.log("new")')
    })
    const expandedPanel = view.getByTestId('workspace-panel')
    expect(expandedPanel.style.width).toBe('860px')
    expect(expandedPanel.style.maxWidth).toBe('min(62%, calc(100% - 328px))')
    expect(expandedPanel.style.minWidth).toBe('min(420px, 54%)')
    expect(view.getAllByText('Diff').length).toBeGreaterThan(0)
  })

  it('renders transcript-derived changed files for non-git sessions', async () => {
    getMocks().getWorkspaceStatusMock.mockResolvedValue({
      state: 'ok',
      workDir: '/tmp/non-git-session',
      repoName: 'non-git-session',
      branch: null,
      isGitRepo: false,
      changedFiles: [
        {
          path: 'src/app.ts',
          status: 'modified',
          additions: 1,
          deletions: 1,
        },
      ],
    })
    getMocks().getWorkspaceDiffMock.mockResolvedValue({
      state: 'ok',
      path: 'src/app.ts',
      diff: 'diff --session a/src/app.ts b/src/app.ts\n-export const answer = 1\n+export const answer = 2',
    })

    await act(() => {
      useWorkspacePanelStore.getState().openPanel('session-non-git')
    })

    const view = await renderPanel('session-non-git')

    await waitFor(() => {
      expect(view.getByText('src/app.ts')).toBeTruthy()
    })
    expect(view.queryByText('No matching files')).toBeNull()

    await clickElement(view.getByText('src/app.ts'))

    await waitFor(() => {
      expect(getMocks().getWorkspaceDiffMock).toHaveBeenCalledWith('session-non-git', 'src/app.ts')
    })
    await waitFor(() => {
      expect(view.getByTestId('workspace-code').textContent).toContain('export const answer = 2')
    })
  })

  it('lazy loads the root tree, expands directories, and opens file previews from the all-files view', async () => {
    const statusRequest = deferred<{
      state: 'ok'
      workDir: string
      repoName: string
      branch: string
      isGitRepo: true
      changedFiles: []
    }>()
    const rootTreeRequest = deferred<{
      state: 'ok'
      path: ''
      entries: Array<{ name: string; path: string; isDirectory: boolean }>
    }>()
    const childTreeRequest = deferred<{
      state: 'ok'
      path: 'src'
      entries: Array<{ name: string; path: string; isDirectory: boolean }>
    }>()
    const fileRequest = deferred<{
      state: 'ok'
      path: string
      content: string
      language: string
      size: number
    }>()

    getMocks().getWorkspaceStatusMock.mockReturnValue(statusRequest.promise)
    getMocks().getWorkspaceTreeMock
      .mockReturnValueOnce(rootTreeRequest.promise)
      .mockReturnValueOnce(childTreeRequest.promise)
    getMocks().getWorkspaceFileMock.mockReturnValue(fileRequest.promise)

    await act(() => {
      useWorkspacePanelStore.getState().openPanel('session-tree')
    })

    const view = await renderPanel('session-tree')

    expect(view.getByRole('button', { name: 'Changed files' })).toBeTruthy()

    await clickElement(view.getByRole('button', { name: 'Changed files' }))
    await clickElement(view.getByRole('menuitem', { name: 'All files' }))

    await waitFor(() => {
      expect(getMocks().getWorkspaceTreeMock).toHaveBeenCalledWith('session-tree', '')
    })

    await act(async () => {
      statusRequest.resolve({
        state: 'ok',
        workDir: '/repo',
        repoName: 'repo',
        branch: 'main',
        isGitRepo: true,
        changedFiles: [],
      })
      rootTreeRequest.resolve({
        state: 'ok',
        path: '',
        entries: [
          { name: 'src', path: 'src', isDirectory: true },
          { name: 'README.md', path: 'README.md', isDirectory: false },
        ],
      })
      await Promise.all([statusRequest.promise, rootTreeRequest.promise])
    })

    const folderLabel = await view.findByText('src')
    const folderButton = folderLabel.closest('button')
    if (!folderButton) {
      throw new Error('Expected src label to be rendered inside a folder button')
    }
    expect(folderButton.getAttribute('aria-expanded')).toBe('false')

    await clickElement(folderButton)

    await waitFor(() => {
      expect(getMocks().getWorkspaceTreeMock).toHaveBeenCalledWith('session-tree', 'src')
    })
    await act(async () => {
      childTreeRequest.resolve({
        state: 'ok',
        path: 'src',
        entries: [{ name: 'index.ts', path: 'src/index.ts', isDirectory: false }],
      })
      await childTreeRequest.promise
    })
    await waitFor(() => {
      expect(folderButton.getAttribute('aria-expanded')).toBe('true')
    })

    await clickElement(await view.findByText('index.ts'))

    await waitFor(() => {
      expect(getMocks().getWorkspaceFileMock).toHaveBeenCalledWith('session-tree', 'src/index.ts')
    })
    await act(async () => {
      fileRequest.resolve({
        state: 'ok',
        path: 'src/index.ts',
        content: 'export const ready = true',
        language: 'typescript',
        size: 25,
      })
      await fileRequest.promise
    })

    await waitFor(() => {
      expect(view.getByTestId('workspace-code').textContent).toContain('export const ready = true')
    })
    expect(view.getAllByText('File').length).toBeGreaterThan(0)
  })

  it('renders multiple preview tabs and closes only the exact requested tab', async () => {
    await setWorkspaceState((state) => ({
      ...state,
      panelBySession: {
        ...state.panelBySession,
        'session-tabs': {
          isOpen: true,
          activeView: 'changed',
        },
      },
      statusBySession: {
        ...state.statusBySession,
        'session-tabs': {
          state: 'ok',
          workDir: '/repo',
          repoName: 'repo',
          branch: 'main',
          isGitRepo: true,
          changedFiles: [],
        },
      },
      previewTabsBySession: {
        ...state.previewTabsBySession,
        'session-tabs': [
          {
            id: 'file:src/a.ts',
            path: 'src/a.ts',
            kind: 'file',
            title: 'a.ts',
            language: 'typescript',
            content: 'export const a = 1',
            state: 'ok',
            size: 18,
          },
          {
            id: 'diff:src/a.ts',
            path: 'src/a.ts',
            kind: 'diff',
            title: 'a.ts',
            diff: '@@ -1 +1 @@',
            state: 'ok',
          },
          {
            id: 'file:src/b.ts',
            path: 'src/b.ts',
            kind: 'file',
            title: 'b.ts',
            language: 'typescript',
            content: 'export const b = 1',
            state: 'ok',
            size: 18,
          },
        ],
      },
      activePreviewTabIdBySession: {
        ...state.activePreviewTabIdBySession,
        'session-tabs': 'diff:src/a.ts',
      },
    }))

    const view = await renderPanel('session-tabs')

    expect(view.getByRole('tablist', { name: 'Preview tabs' })).toBeTruthy()
    expect(view.getAllByRole('tab', { name: /a\.ts/ })).toHaveLength(2)
    expect(view.getAllByText('a.ts').length).toBeGreaterThanOrEqual(2)
    expect(view.getAllByText('b.ts').length).toBeGreaterThanOrEqual(1)

    await clickElement(view.getByLabelText('Close tab a.ts Diff'))

    expect(view.queryByLabelText('Close tab a.ts Diff')).toBeNull()
    expect(view.getByLabelText('Close tab a.ts File')).toBeTruthy()
    expect(view.getAllByText('b.ts').length).toBeGreaterThanOrEqual(1)
  })

  it('caps rendered preview lines to keep large diffs responsive', async () => {
    const longDiff = Array.from({ length: 650 }, (_, index) => `+line ${index + 1}`).join('\n')

    await setWorkspaceState((state) => ({
      ...state,
      panelBySession: {
        ...state.panelBySession,
        'session-large-preview': {
          isOpen: true,
          activeView: 'changed',
        },
      },
      statusBySession: {
        ...state.statusBySession,
        'session-large-preview': {
          state: 'ok',
          workDir: '/repo',
          repoName: 'repo',
          branch: 'main',
          isGitRepo: true,
          changedFiles: [],
        },
      },
      previewTabsBySession: {
        ...state.previewTabsBySession,
        'session-large-preview': [{
          id: 'diff:large.ts',
          path: 'large.ts',
          kind: 'diff',
          title: 'large.ts',
          diff: longDiff,
          state: 'ok',
        }],
      },
      activePreviewTabIdBySession: {
        ...state.activePreviewTabIdBySession,
        'session-large-preview': 'diff:large.ts',
      },
    }))

    const view = await renderPanel('session-large-preview')
    const highlightedCode = view.getByTestId('workspace-code').textContent ?? ''

    expect(highlightedCode).toContain('+line 1')
    expect(highlightedCode).toContain('+line 420')
    expect(highlightedCode).not.toContain('+line 421')
    expect(view.getByText('Showing first 420 lines. Open in your editor for the full file.')).toBeTruthy()
  })

  it('renders image previews from workspace files', async () => {
    await setWorkspaceState((state) => ({
      ...state,
      panelBySession: {
        ...state.panelBySession,
        'session-image-preview': {
          isOpen: true,
          activeView: 'all',
        },
      },
      treeBySessionPath: {
        ...state.treeBySessionPath,
        'session-image-preview': {
          '': {
            state: 'ok',
            path: '',
            entries: [{ name: 'logo.png', path: 'logo.png', isDirectory: false }],
          },
        },
      },
    }))

    getMocks().getWorkspaceFileMock.mockResolvedValue({
      state: 'ok',
      path: 'logo.png',
      previewType: 'image',
      dataUrl: 'data:image/png;base64,iVBORw0KGgo=',
      mimeType: 'image/png',
      language: 'image',
      size: 8,
    })

    const view = await renderPanel('session-image-preview')

    await clickElement(await view.findByText('logo.png'))

    const image = await view.findByRole('img', { name: 'logo.png' })
    expect(image.getAttribute('src')).toBe('data:image/png;base64,iVBORw0KGgo=')
  })

  it('renders markdown file previews as formatted documents', async () => {
    await setWorkspaceState((state) => ({
      ...state,
      panelBySession: {
        ...state.panelBySession,
        'session-markdown-preview': {
          isOpen: true,
          activeView: 'all',
        },
      },
      statusBySession: {
        ...state.statusBySession,
        'session-markdown-preview': {
          state: 'ok',
          workDir: '/repo',
          repoName: 'repo',
          branch: 'main',
          isGitRepo: true,
          changedFiles: [],
        },
      },
      previewTabsBySession: {
        ...state.previewTabsBySession,
        'session-markdown-preview': [{
          id: 'file:README.md',
          path: 'README.md',
          kind: 'file',
          title: 'README.md',
          language: 'markdown',
          content: '# Project Notes\n\n- **Done** item\n\n```ts\nexport const ok = true\n```',
          state: 'ok',
          size: 70,
        }],
      },
      activePreviewTabIdBySession: {
        ...state.activePreviewTabIdBySession,
        'session-markdown-preview': 'file:README.md',
      },
    }))

    const view = await renderPanel('session-markdown-preview')

    expect(view.getByRole('heading', { name: 'Project Notes', level: 1 })).toBeTruthy()
    expect(view.getByText('Done')).toBeTruthy()
    expect(view.container.textContent).toContain('export const ok = true')
    expect(view.queryByTestId('workspace-code')).toBeNull()
  })

  it('uses the localized view menu label', async () => {
    await setSettingsState({ ...settingsInitialState, locale: 'zh' })
    await setWorkspaceState((state) => ({
      ...state,
      panelBySession: {
        ...state.panelBySession,
        'session-zh': {
          isOpen: true,
          activeView: 'changed',
        },
      },
      statusBySession: {
        ...state.statusBySession,
        'session-zh': {
          state: 'ok',
          workDir: '/repo',
          repoName: 'repo',
          branch: 'main',
          isGitRepo: true,
          changedFiles: [],
        },
      },
    }))

    const view = await renderPanel('session-zh')

    expect(view.getByRole('button', { name: '已更改文件' })).toBeTruthy()
  })

  it('shows explicit empty and error states in the changed view', async () => {
    await setWorkspaceState((state) => ({
      ...state,
      panelBySession: {
        ...state.panelBySession,
        'session-empty': {
          isOpen: true,
          activeView: 'changed',
        },
      },
      statusBySession: {
        ...state.statusBySession,
        'session-empty': {
          state: 'ok',
          workDir: '/repo',
          repoName: 'repo',
          branch: 'main',
          isGitRepo: true,
          changedFiles: [],
        },
      },
    }))

    const view = await renderPanel('session-empty')

    expect(view.getByText('No changes')).toBeTruthy()

    await setWorkspaceState((state) => ({
      ...state,
      panelBySession: {
        ...state.panelBySession,
        'session-error': {
          isOpen: true,
          activeView: 'changed',
        },
      },
      errors: {
        ...state.errors,
        statusBySession: {
          ...state.errors.statusBySession,
          'session-error': 'status failed',
        },
      },
    }))

    await act(() => {
      view.rerender(<WorkspacePanel sessionId="session-error" />)
    })

    expect(view.getByText('status failed')).toBeTruthy()
  })
})
