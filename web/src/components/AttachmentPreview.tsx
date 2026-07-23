import { AlertTriangle, File, FileAudio, FileVideo, ImageOff } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import type { ArchiveAttachment, MediaDerivative } from '../lib/contracts'
import { formatSeconds, sentenceCase } from '../lib/format'

type PreviewAttachment = ArchiveAttachment & { type?: string }
type PreviewDerivativeKind = MediaDerivative['kind']

const derivativeHrefPattern = /^\/api\/v1\/derivatives\/[0-9a-f]{64}$/
const blobHrefPattern = /^\/api\/v1\/blobs\/[0-9a-f]{64}$/

function readyDerivative(
  item: PreviewAttachment,
  kind: PreviewDerivativeKind,
): MediaDerivative | undefined {
  return item.derivatives?.find((candidate) => (
    candidate.kind === kind
    && candidate.status === 'ready'
    && typeof candidate.href === 'string'
    && derivativeHrefPattern.test(candidate.href)
  ))
}

function attachmentClass(item: PreviewAttachment): string {
  if (item.type) return item.type
  const kind = item.kind.toLowerCase()
  const mediaType = item.media_type?.toLowerCase() ?? ''
  // Facebook voice messages are commonly stored in MP4 containers. The explicit source kind
  // therefore has to win over the container MIME type or an audio message is rendered as video.
  if (kind.includes('gif')) return 'gif'
  if (kind.includes('sticker')) return 'sticker'
  if (kind.includes('photo') || kind === 'thread_image') return 'photo'
  if (kind.includes('audio')) return 'audio'
  if (kind.includes('video')) return 'video'
  if (kind.includes('file') || kind.includes('document')) return 'document'
  if (mediaType.startsWith('image/')) return 'photo'
  if (mediaType.startsWith('audio/')) return 'audio'
  if (mediaType.startsWith('video/')) return 'video'
  return 'other'
}

function Placeholder({ item, failed = false }: { item: PreviewAttachment; failed?: boolean }) {
  const Icon = failed ? AlertTriangle : item.availability === 'missing' ? ImageOff : File
  const label = failed
    ? 'Preview unavailable'
    : item.availability === 'missing'
      ? 'File missing from export'
      : item.is_external_reference
        ? 'External reference only'
        : 'No local preview'
  return (
    <div className="attachment-placeholder" role="img" aria-label={`${item.filename ?? 'Attachment'}: ${label}`}>
      <Icon size={25} aria-hidden="true" />
      <strong>{label}</strong>
      <span>{sentenceCase(item.availability)}</span>
    </div>
  )
}

function DurationLabel({ durationMs }: { durationMs?: number | null }) {
  if (durationMs == null || !Number.isFinite(durationMs) || durationMs < 0) return null
  const label = formatSeconds(durationMs / 1000)
  return <span className="attachment-duration"><span className="sr-only">Duration </span>{label}</span>
}

export function AttachmentPreview({
  item,
  compact = false,
  seekMs,
}: {
  item: PreviewAttachment
  compact?: boolean
  seekMs?: number
}) {
  const [failedSources, setFailedSources] = useState<string[]>([])
  const audioRef = useRef<HTMLAudioElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const type = attachmentClass(item)
  const source = typeof item.blob_href === 'string' && blobHrefPattern.test(item.blob_href)
    ? item.blob_href
    : undefined
  const retainedLocally = item.availability !== 'missing' && !item.is_external_reference
  const thumbnail = readyDerivative(item, 'image_thumbnail')
  const poster = readyDerivative(item, 'video_poster')
  const waveform = readyDerivative(item, 'audio_waveform')
  const thumbnailHref = thumbnail?.href ?? undefined
  const posterHref = poster?.href ?? undefined
  const waveformHref = waveform?.href ?? undefined
  const hasFailed = useCallback(
    (href: string | undefined) => href == null || failedSources.includes(href),
    [failedSources],
  )
  const markFailed = useCallback((href: string | undefined) => {
    if (!href) return
    setFailedSources((current) => current.includes(href) ? current : [...current, href])
  }, [])
  const seekToRequestedTime = useCallback(() => {
    const player = type === 'audio' ? audioRef.current : videoRef.current
    if (!player || seekMs == null || !Number.isFinite(seekMs) || seekMs < 0) return
    try {
      player.currentTime = seekMs / 1000
    } catch {
      // Some browser media backends reject seeking until metadata is available. The
      // loaded-metadata callback below retries without turning a valid file into an error state.
    }
  }, [seekMs, type])

  useEffect(seekToRequestedTime, [seekToRequestedTime, source])

  if (!retainedLocally) return <Placeholder item={item} />

  if (type === 'photo' || type === 'gif' || type === 'sticker') {
    const previewSource = !hasFailed(thumbnailHref) ? thumbnailHref : !hasFailed(source) ? source : undefined
    if (!previewSource) return <Placeholder item={item} failed={Boolean(source || thumbnailHref)} />
    return (
      <img
        className={`attachment-preview attachment-preview--image ${compact ? 'attachment-preview--compact' : ''}`}
        src={previewSource}
        alt={item.filename || `${sentenceCase(type)} attachment`}
        loading="lazy"
        decoding="async"
        onError={() => markFailed(previewSource)}
      />
    )
  }

  if (type === 'video') {
    if (hasFailed(source)) return <Placeholder item={item} failed={Boolean(source)} />
    return (
      <div className="attachment-player">
        <FileVideo size={19} aria-hidden="true" />
        <video
          ref={videoRef}
          className={`attachment-preview attachment-preview--video ${compact ? 'attachment-preview--compact' : ''}`}
          src={source}
          poster={!hasFailed(posterHref) ? posterHref : undefined}
          controls
          preload="metadata"
          data-seek-ms={seekMs}
          onLoadedMetadata={seekToRequestedTime}
          onCanPlay={seekToRequestedTime}
          onError={() => markFailed(source)}
        >
          This browser cannot play the archived video.
        </video>
        <DurationLabel durationMs={poster?.duration_ms} />
      </div>
    )
  }

  if (type === 'audio') {
    if (hasFailed(source)) return <Placeholder item={item} failed={Boolean(source)} />
    return (
      <div className="attachment-player attachment-player--audio">
        <FileAudio size={19} aria-hidden="true" />
        {!hasFailed(waveformHref) ? (
          <img
            className="attachment-waveform"
            src={waveformHref}
            alt=""
            aria-hidden="true"
            loading="lazy"
            decoding="async"
            onError={() => markFailed(waveformHref)}
          />
        ) : null}
        <audio
          ref={audioRef}
          src={source}
          controls
          preload="metadata"
          data-seek-ms={seekMs}
          onLoadedMetadata={seekToRequestedTime}
          onCanPlay={seekToRequestedTime}
          onError={() => markFailed(source)}
        >
          This browser cannot play the archived audio.
        </audio>
        <DurationLabel durationMs={waveform?.duration_ms} />
      </div>
    )
  }

  if (hasFailed(source)) return <Placeholder item={item} failed={Boolean(source)} />
  return (
    <a className="document-preview" href={source} target="_blank" rel="noreferrer">
      <File size={24} aria-hidden="true" />
      <span>
        <strong>{item.filename || 'Archived file'}</strong>
        <small>Open file safely</small>
      </span>
    </a>
  )
}
