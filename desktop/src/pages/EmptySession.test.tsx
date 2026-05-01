import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom'

const mocks = vi.hoisted(() => ({
  createSession: vi.fn(),
  listSessions: vi.fn(),
  getMessages: vi.fn(),
  getSlashCommands: vi.fn(),
  listSkills: vi.fn(),
  getTasksForList: vi.fn(),
  resetTaskList: vi.fn(),
  wsClearHandlers: vi.fn(),
  wsConnect: vi.fn(),
  wsOnMessage: vi.fn(),
  wsSend: vi.fn(),
  wsDisconnect: vi.fn(),
}))

vi.mock('../api/sessions', () => ({
  sessionsApi: {
    create: mocks.createSession,
    list: mocks.listSessions,
    getMessages: mocks.getMessages,
    getSlashCommands: mocks.getSlashCommands,
  },
}))

vi.mock('../api/skills', () => ({
  skillsApi: {
    list: mocks.listSkills,
  },
}))

vi.mock('../api/cliTasks', () => ({
  cliTasksApi: {
    getTasksForList: mocks.getTasksForList,
    resetTaskList: mocks.resetTaskList,
  },
}))

vi.mock('../api/websocket', () => ({
  wsManager: {
    clearHandlers: mocks.wsClearHandlers,
    connect: mocks.wsConnect,
    onMessage: mocks.wsOnMessage,
    send: mocks.wsSend,
    disconnect: mocks.wsDisconnect,
  },
}))

vi.mock('../components/shared/DirectoryPicker', () => ({
  DirectoryPicker: ({ value, onChange }: { value: string; onChange: (path: string) => void }) => (
    <button type="button" aria-label="Pick project" data-value={value} onClick={() => onChange('/workspace/project')}>
      Pick project
    </button>
  ),
}))

vi.mock('../components/controls/PermissionModeSelector', () => ({
  PermissionModeSelector: () => <button type="button">Bypass</button>,
}))

vi.mock('../components/controls/ModelSelector', () => ({
  ModelSelector: () => <button type="button">Model</button>,
}))

import { EmptySession } from './EmptySession'
import { useChatStore } from '../stores/chatStore'
import { useSessionRuntimeStore } from '../stores/sessionRuntimeStore'
import { useSessionStore } from '../stores/sessionStore'
import { useSettingsStore } from '../stores/settingsStore'
import { useTabStore } from '../stores/tabStore'
import { useUIStore } from '../stores/uiStore'

describe('EmptySession', () => {
  const initialSessionState = useSessionStore.getInitialState()
  const initialChatState = useChatStore.getInitialState()
  const initialTabState = useTabStore.getInitialState()
  const initialRuntimeState = useSessionRuntimeStore.getInitialState()
  const initialUiState = useUIStore.getInitialState()

  beforeEach(() => {
    vi.clearAllMocks()
    useSettingsStore.setState({ locale: 'en', activeProviderName: null })
    useSessionStore.setState(initialSessionState, true)
    useChatStore.setState(initialChatState, true)
    useTabStore.setState(initialTabState, true)
    useSessionRuntimeStore.setState(initialRuntimeState, true)
    useUIStore.setState(initialUiState, true)

    mocks.createSession.mockResolvedValue({ sessionId: 'draft-session' })
    mocks.listSessions.mockResolvedValue({
      sessions: [{
        id: 'draft-session',
        title: 'New Session',
        createdAt: '2026-05-01T00:00:00.000Z',
        modifiedAt: '2026-05-01T00:00:00.000Z',
        messageCount: 0,
        projectPath: '/workspace/project',
        workDir: '/workspace/project',
        workDirExists: true,
      }],
      total: 1,
    })
    mocks.getMessages.mockResolvedValue({ messages: [] })
    mocks.getSlashCommands.mockResolvedValue({ commands: [] })
    mocks.listSkills.mockResolvedValue({ skills: [] })
    mocks.getTasksForList.mockResolvedValue({ tasks: [] })
    mocks.resetTaskList.mockResolvedValue(undefined)
  })

  afterEach(() => {
    useSessionStore.setState(initialSessionState, true)
    useChatStore.setState(initialChatState, true)
    useTabStore.setState(initialTabState, true)
    useSessionRuntimeStore.setState(initialRuntimeState, true)
    useUIStore.setState(initialUiState, true)
  })

  it('creates an empty session as soon as a project is selected', async () => {
    render(<EmptySession />)

    fireEvent.change(screen.getByRole('textbox'), {
      target: { value: 'draft question', selectionStart: 14 },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Pick project' }))

    await waitFor(() => {
      expect(mocks.createSession).toHaveBeenCalledWith('/workspace/project')
    })

    expect(useTabStore.getState().activeTabId).toBe('draft-session')
    expect(useTabStore.getState().tabs).toEqual([
      { sessionId: 'draft-session', title: 'New Session', type: 'session', status: 'idle' },
    ])
    expect(useSessionStore.getState().sessions[0]).toMatchObject({
      id: 'draft-session',
      workDir: '/workspace/project',
    })
    expect(useChatStore.getState().sessions['draft-session']?.composerPrefill?.text).toBe('draft question')
    expect(mocks.wsConnect).toHaveBeenCalledWith('draft-session')
  })
})
