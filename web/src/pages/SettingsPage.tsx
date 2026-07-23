import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Check,
  Cloud,
  CloudOff,
  FolderLock,
  ImageOff,
  KeyRound,
  LockKeyhole,
  MapPin,
  Save,
  ShieldCheck,
  UserRound,
} from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { api } from '../lib/api'
import type { SettingsResponse } from '../lib/contracts'
import { sentenceCase } from '../lib/format'
import { ErrorState, PageHeading, PageLoading } from '../components/ui'

const commonTimezones = [
  'America/New_York',
  'America/Chicago',
  'America/Denver',
  'America/Los_Angeles',
  'America/Phoenix',
  'UTC',
  'Europe/London',
  'Europe/Berlin',
  'Asia/Tokyo',
  'Australia/Sydney',
]

function isValidTimezone(timezone: string): boolean {
  try {
    new Intl.DateTimeFormat('en-US', { timeZone: timezone }).format()
    return true
  } catch {
    return false
  }
}

export function SettingsPage() {
  const queryClient = useQueryClient()
  const settings = useQuery({ queryKey: ['settings'], queryFn: api.settings })
  const [timezone, setTimezone] = useState('America/New_York')
  const [cloudEnabled, setCloudEnabled] = useState(false)

  useEffect(() => {
    if (!settings.data) return
    setTimezone(settings.data.timezone ?? 'America/New_York')
    setCloudEnabled(settings.data.cloud_enabled === true)
  }, [settings.data])

  const mutation = useMutation({
    mutationFn: api.updateSettings,
    onSuccess: (updated) => {
      queryClient.setQueryData<SettingsResponse>(['settings'], updated)
    },
  })

  const initialTimezone = settings.data?.timezone ?? 'America/New_York'
  const initialCloud = settings.data?.cloud_enabled === true
  const dirty = timezone !== initialTimezone || cloudEnabled !== initialCloud
  const timezoneValid = useMemo(() => isValidTimezone(timezone), [timezone])

  if (settings.isLoading) return <PageLoading label="Loading local settings" />
  if (settings.isError) return <ErrorState error={settings.error} onRetry={() => settings.refetch()} />

  const save = (event: React.FormEvent) => {
    event.preventDefault()
    if (!timezoneValid || !dirty) return
    mutation.mutate({ timezone, cloud_enabled: cloudEnabled })
  }

  return (
    <div className="page-stack">
      <PageHeading
        eyebrow="Local preferences"
        title="Settings"
        description="Configure display behavior and explicit privacy choices. Source identity and archive location remain read-only here."
      />

      <form className="settings-layout" onSubmit={save}>
        <div className="settings-main">
          <section className="panel settings-section" aria-labelledby="display-settings-heading">
            <div className="settings-section__header">
              <span aria-hidden="true"><MapPin size={20} /></span>
              <div>
                <h2 id="display-settings-heading">Display</h2>
                <p>Dates remain stored in UTC and are converted only for presentation.</p>
              </div>
            </div>
            <label className="field-control">
              <span>Display timezone</span>
              <input
                type="text"
                list="timezone-options"
                value={timezone}
                onChange={(event) => setTimezone(event.target.value)}
                aria-invalid={!timezoneValid}
                aria-describedby="timezone-help timezone-error"
                autoComplete="off"
              />
              <datalist id="timezone-options">
                {commonTimezones.map((zone) => <option key={zone} value={zone} />)}
              </datalist>
              <small id="timezone-help">Enter an IANA timezone such as America/New_York.</small>
              {!timezoneValid ? <strong id="timezone-error" className="field-error">Enter a valid IANA timezone.</strong> : null}
            </label>
          </section>

          <section className="panel settings-section" aria-labelledby="privacy-settings-heading">
            <div className="settings-section__header">
              <span aria-hidden="true"><ShieldCheck size={20} /></span>
              <div>
                <h2 id="privacy-settings-heading">Privacy & processing</h2>
                <p>Cloud use is always explicit and permanently visible in the application shell.</p>
              </div>
            </div>

            <label className={`toggle-row ${cloudEnabled ? 'toggle-row--warning' : ''}`}>
              <span className="toggle-row__icon" aria-hidden="true">
                {cloudEnabled ? <Cloud size={20} /> : <CloudOff size={20} />}
              </span>
              <span className="toggle-row__copy">
                <strong>Allow cloud-assisted features</strong>
                <small>
                  {cloudEnabled
                    ? 'Only a question and the locally retrieved excerpts needed for a requested RAG answer may be sent.'
                    : 'All current archive processing stays on this machine.'}
                </small>
              </span>
              <input
                className="toggle-input"
                type="checkbox"
                checked={cloudEnabled}
                onChange={(event) => setCloudEnabled(event.target.checked)}
                aria-label="Allow cloud-assisted features"
              />
              <span className="toggle-visual" aria-hidden="true"><span /></span>
            </label>

            {cloudEnabled ? (
              <div className="inline-alert inline-alert--warning" role="status">
                <Cloud size={17} aria-hidden="true" />
                <div>
                  <strong>{initialCloud ? 'Cloud-assisted RAG is enabled.' : 'Cloud access will be enabled after you save.'}</strong>
                  <span>This setting alone uploads nothing. A call occurs only when you request a cited answer; every call is source-limited and locally audited.</span>
                </div>
              </div>
            ) : (
              <div className="inline-alert inline-alert--success">
                <LockKeyhole size={17} aria-hidden="true" />
                <div>
                  <strong>Local-only mode is active.</strong>
                  <span>The browser communicates with the localhost service only.</span>
                </div>
              </div>
            )}

            <div className="inline-alert" role="note">
              <ShieldCheck size={17} aria-hidden="true" />
              <div>
                <strong>OpenAI API data controls</strong>
                <span>RAG requests use the Responses API with storage disabled. API data is not used to train OpenAI models by default, but standard abuse-monitoring logs may retain content for up to 30 days unless your OpenAI project has stricter approved controls.</span>
              </div>
            </div>

            <div className="gated-setting" role="group" aria-labelledby="image-stage-title">
              <span className="gated-setting__icon" aria-hidden="true"><ImageOff size={20} /></span>
              <div>
                <div className="gated-setting__title">
                  <strong id="image-stage-title">Image semantic classification</strong>
                  <span>Stage 6</span>
                </div>
                <p>Disabled until every earlier stage is complete and you give a new, direct approval.</p>
              </div>
              <button className="button button--secondary" type="button" disabled>
                <KeyRound size={15} aria-hidden="true" />
                Approval required
              </button>
            </div>
          </section>
        </div>

        <aside className="settings-sidebar" aria-label="Archive identity">
          <section className="panel settings-section settings-section--compact">
            <div className="settings-section__header">
              <span aria-hidden="true"><FolderLock size={20} /></span>
              <div>
                <h2>Archive identity</h2>
                <p>Set by the verified local catalog.</p>
              </div>
            </div>
            <dl className="settings-facts">
              <div>
                <dt><UserRound size={15} aria-hidden="true" /> Owner identity</dt>
                <dd>{settings.data?.owner_identity ?? 'Not configured'}</dd>
              </div>
              <div>
                <dt><FolderLock size={15} aria-hidden="true" /> Archive root</dt>
                <dd title={settings.data?.archive_root_display}>{settings.data?.archive_root_display ?? 'Not reported'}</dd>
              </div>
              <div>
                <dt><LockKeyhole size={15} aria-hidden="true" /> Access mode</dt>
                <dd>{sentenceCase(settings.data?.access_mode)}</dd>
              </div>
            </dl>
          </section>

          <div className="settings-savebar">
            <div className="settings-savebar__status" aria-live="polite">
              {mutation.isSuccess && !dirty ? (
                <><Check size={15} aria-hidden="true" /> Saved</>
              ) : dirty ? (
                'Unsaved changes'
              ) : (
                'Settings are up to date'
              )}
            </div>
            {mutation.isError ? (
              <p className="field-error" role="alert">
                {mutation.error instanceof Error ? mutation.error.message : 'Settings could not be saved.'}
              </p>
            ) : null}
            <button
              className="button button--primary button--full"
              type="submit"
              disabled={!dirty || !timezoneValid || mutation.isPending}
            >
              <Save size={16} aria-hidden="true" />
              {mutation.isPending ? 'Saving…' : 'Save settings'}
            </button>
          </div>
        </aside>
      </form>
    </div>
  )
}
