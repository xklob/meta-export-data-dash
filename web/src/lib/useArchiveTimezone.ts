import { useQuery } from '@tanstack/react-query'
import { api } from './api'

const DEFAULT_TIMEZONE = 'America/New_York'

export function useArchiveTimezone(): string {
  const settings = useQuery({
    queryKey: ['settings'],
    queryFn: api.settings,
    staleTime: 30_000,
    retry: 1,
  })
  return settings.data?.timezone || DEFAULT_TIMEZONE
}
