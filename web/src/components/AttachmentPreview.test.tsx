import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { ArchiveAttachment, MediaDerivative } from '../lib/contracts'
import { AttachmentPreview } from './AttachmentPreview'

function derivative(
  kind: MediaDerivative['kind'],
  idCharacter: string,
  durationMs?: number,
): MediaDerivative {
  const id = idCharacter.repeat(64)
  return {
    id,
    kind,
    status: 'ready',
    media_type: kind === 'audio_waveform' ? 'image/png' : 'image/jpeg',
    duration_ms: durationMs,
    href: `/api/v1/derivatives/${id}`,
  }
}

function attachment(overrides: Partial<ArchiveAttachment> = {}): ArchiveAttachment {
  return {
    id: 'attachment-one',
    kind: 'photos',
    filename: 'retained.jpg',
    media_type: 'image/jpeg',
    availability: 'resolved',
    blob_href: `/api/v1/blobs/${'f'.repeat(64)}`,
    is_external_reference: false,
    ...overrides,
  }
}

describe('AttachmentPreview derivatives', () => {
  it('uses a safe generated thumbnail and falls back to the original blob', () => {
    const thumbnail = derivative('image_thumbnail', 'a')
    render(<AttachmentPreview item={attachment({ derivatives: [thumbnail] })} />)

    const image = screen.getByRole('img', { name: 'retained.jpg' })
    expect(image).toHaveAttribute('src', thumbnail.href)

    fireEvent.error(image)
    expect(screen.getByRole('img', { name: 'retained.jpg' })).toHaveAttribute(
      'src',
      `/api/v1/blobs/${'f'.repeat(64)}`,
    )

    fireEvent.error(screen.getByRole('img', { name: 'retained.jpg' }))
    expect(screen.getByText('Preview unavailable')).toBeVisible()
  })

  it('keeps video playback on the original blob while using its poster and duration', () => {
    const poster = derivative('video_poster', 'b', 125_500)
    const { container } = render(<AttachmentPreview item={attachment({
      kind: 'videos',
      filename: 'retained.mp4',
      media_type: 'video/mp4',
      derivatives: [poster],
    })} />)

    const video = container.querySelector('video')
    expect(video).toHaveAttribute('src', `/api/v1/blobs/${'f'.repeat(64)}`)
    expect(video).toHaveAttribute('poster', poster.href)
    expect(screen.getByText((_content, element) => (
      element?.classList.contains('attachment-duration') === true
      && element.textContent === 'Duration 2m 5s'
    ))).toBeVisible()
  })

  it('renders an audio waveform without sacrificing playback and hides a failed waveform', () => {
    const waveform = derivative('audio_waveform', 'c', 12_345)
    const { container } = render(<AttachmentPreview item={attachment({
      kind: 'audio_files',
      filename: 'voice-message.mp4',
      media_type: 'video/mp4',
      derivatives: [waveform],
    })} />)

    const audio = container.querySelector('audio')
    const waveformImage = container.querySelector('img.attachment-waveform')
    expect(audio).toHaveAttribute('src', `/api/v1/blobs/${'f'.repeat(64)}`)
    expect(waveformImage).toHaveAttribute('src', waveform.href)
    expect(screen.getByText((_content, element) => (
      element?.classList.contains('attachment-duration') === true
      && element.textContent === 'Duration 12s'
    ))).toBeVisible()

    fireEvent.error(waveformImage!)
    expect(container.querySelector('img.attachment-waveform')).not.toBeInTheDocument()
    expect(container.querySelector('audio')).toHaveAttribute('src', `/api/v1/blobs/${'f'.repeat(64)}`)
  })

  it('rejects derivative URLs outside the fixed authenticated route', () => {
    const unsafe = {
      ...derivative('image_thumbnail', 'd'),
      href: 'https://example.invalid/tracker.jpg',
    }
    render(<AttachmentPreview item={attachment({ derivatives: [unsafe] })} />)

    expect(screen.getByRole('img', { name: 'retained.jpg' })).toHaveAttribute(
      'src',
      `/api/v1/blobs/${'f'.repeat(64)}`,
    )
  })

  it('never auto-fetches a blob URL outside the fixed authenticated route', () => {
    const { container } = render(<AttachmentPreview item={attachment({
      blob_href: 'https://example.invalid/private-tracker.jpg',
    })} />)

    expect(container.querySelector('img')).not.toBeInTheDocument()
    expect(container.querySelector('video')).not.toBeInTheDocument()
    expect(container.querySelector('audio')).not.toBeInTheDocument()
    expect(container.querySelector('a')).not.toBeInTheDocument()
    expect(screen.getByText('No local preview')).toBeVisible()
  })
})
