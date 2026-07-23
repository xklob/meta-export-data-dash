import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { expect, it, vi } from 'vitest'
import { CursorWindowControls } from './CursorWindowControls'

const baseProps = {
  label: 'Fixture results',
  pageNumber: 1,
  itemCount: 1,
  total: 300,
  hasPreviousPage: false,
  hasNextPage: true,
  loading: false,
  onPreviousPage: vi.fn(),
  onNextPage: vi.fn(),
}

it('keeps unavailable controls discoverable while guarding their handlers', async () => {
  const user = userEvent.setup()
  const onPreviousPage = vi.fn()
  const onNextPage = vi.fn()
  render(
    <CursorWindowControls
      {...baseProps}
      onPreviousPage={onPreviousPage}
      onNextPage={onNextPage}
    />,
  )

  const previous = screen.getByRole('button', { name: 'Previous page' })
  const next = screen.getByRole('button', { name: 'Next page' })
  expect(previous).toHaveAttribute('aria-disabled', 'true')
  expect(previous).not.toBeDisabled()
  await user.click(previous)
  expect(onPreviousPage).not.toHaveBeenCalled()

  expect(next).toHaveAttribute('aria-disabled', 'false')
  await user.click(next)
  expect(onNextPage).toHaveBeenCalledOnce()
})

it('retains focus and announces the requested page across loading, failure, and back recovery', async () => {
  const onPreviousPage = vi.fn()
  const onNextPage = vi.fn()
  const { rerender } = render(
    <CursorWindowControls
      {...baseProps}
      onPreviousPage={onPreviousPage}
      onNextPage={onNextPage}
    />,
  )
  const nav = screen.getByRole('navigation', { name: 'Fixture results pages' })
  const next = screen.getByRole('button', { name: 'Next page' })
  next.focus()

  rerender(
    <CursorWindowControls
      {...baseProps}
      pageNumber={2}
      hasPreviousPage
      loading
      onPreviousPage={onPreviousPage}
      onNextPage={onNextPage}
    />,
  )
  expect(next).toHaveFocus()
  expect(nav).toHaveAttribute('aria-busy', 'true')
  expect(nav).toHaveTextContent('Loading page 2')
  expect(nav).toHaveTextContent('1 current results remain shown · 300 total')
  expect(screen.getByRole('status')).toHaveTextContent(
    'Loading page 2. 1 current results remain shown · 300 total.',
  )
  expect(next).toHaveAttribute('aria-disabled', 'true')
  next.click()
  expect(onNextPage).not.toHaveBeenCalled()

  rerender(
    <CursorWindowControls
      {...baseProps}
      pageNumber={2}
      hasPreviousPage
      onPreviousPage={onPreviousPage}
      onNextPage={onNextPage}
    />,
  )
  expect(next).toHaveFocus()
  expect(nav).toHaveAttribute('aria-busy', 'false')
  expect(nav).toHaveTextContent('Page 2')
  expect(screen.getByRole('status')).toBeEmptyDOMElement()

  rerender(
    <CursorWindowControls
      {...baseProps}
      pageNumber={3}
      itemCount={0}
      total={null}
      hasPreviousPage
      hasNextPage={false}
      onPreviousPage={onPreviousPage}
      onNextPage={onNextPage}
    />,
  )
  expect(next).toHaveFocus()
  expect(next).toHaveAttribute('aria-disabled', 'true')
  expect(nav).toHaveTextContent('Page 3')
  expect(nav).toHaveTextContent('0 shown · total unavailable')

  const previous = screen.getByRole('button', { name: 'Previous page' })
  previous.focus()
  rerender(
    <CursorWindowControls
      {...baseProps}
      pageNumber={2}
      itemCount={0}
      total={null}
      hasPreviousPage
      hasNextPage={false}
      loading
      onPreviousPage={onPreviousPage}
      onNextPage={onNextPage}
    />,
  )
  expect(previous).toHaveFocus()
  expect(nav).toHaveTextContent('Loading page 2')

  rerender(
    <CursorWindowControls
      {...baseProps}
      pageNumber={2}
      hasPreviousPage
      onPreviousPage={onPreviousPage}
      onNextPage={onNextPage}
    />,
  )
  expect(previous).toHaveFocus()
  expect(nav).toHaveTextContent('Page 2')
})
