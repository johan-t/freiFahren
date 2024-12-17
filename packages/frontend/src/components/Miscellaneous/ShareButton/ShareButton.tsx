import React, { useCallback } from 'react'

import { useTranslation } from 'react-i18next'
import { sendAnalyticsEvent } from 'src/hooks/useAnalytics'
import { Report } from 'src/utils/types'

import './ShareButton.css'
import i18next from 'i18next'

interface ShareButtonProps {
    report?: Report
}

const formatTime = (timestamp: string, isHistoric: boolean): string => {
    const date = new Date(timestamp)
    const { t } = useTranslation()

    // For historic reports, show the full date and time
    if (isHistoric) {
        return date.toLocaleString(i18next.language === 'de' ? 'de-DE' : 'en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
        })
    }

    // For recent reports, show relative time
    const now = new Date()
    const diffInMinutes = Math.floor((now.getTime() - date.getTime()) / (1000 * 60))

    if (diffInMinutes < 1) {
        return t('MarkerModal.now')
    }

    if (diffInMinutes < 60) {
        return `${diffInMinutes} ${t('MarkerModal.minutes')} ${t('MarkerModal.ago')}`
    }

    if (diffInMinutes < 1440) {
        // less than 24 hours
        const hours = Math.floor(diffInMinutes / 60)
        if (hours === 1) {
            return t('MarkerModal.oneHourAgo')
        }
        return `${hours} ${t('MarkerModal.hours')} ${t('MarkerModal.ago')}`
    }

    // If more than 24 hours, show the date and time
    return date.toLocaleString(i18next.language === 'de' ? 'de-DE' : 'en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    })
}

const ShareButton: React.FC<ShareButtonProps> = ({ report }) => {
    const { t } = useTranslation()

    const handleShare = useCallback(
        async (e: React.MouseEvent) => {
            e.preventDefault()

            try {
                if (!report) {
                    // Just share the page URL if no report
                    if (navigator.share) {
                        await navigator.share({
                            title: t('Share.title'),
                            url: window.location.href,
                        })
                        await sendAnalyticsEvent('Page Shared', {})
                    } else {
                        await navigator.clipboard.writeText(window.location.href)
                        alert(t('Share.copied'))
                    }
                    return
                }

                const shareText = t('Share.text', {
                    station: report.station.name,
                    direction: report.direction?.name || '?',
                    line: report.line || '?',
                    time: formatTime(report.timestamp, report.isHistoric),
                })

                if (navigator.share) {
                    await navigator.share({
                        title: t('Share.title'),
                        text: shareText,
                        url: window.location.href,
                    })
                    await sendAnalyticsEvent('Marker Shared', {
                        meta: {
                            station: report.station.name,
                            line: report.line,
                            direction: report.direction?.name,
                        },
                    })
                } else {
                    await navigator.clipboard.writeText(`${shareText} ${window.location.href}`)
                    alert(t('Share.copied'))
                }
            } catch (error) {
                console.error('Error sharing:', error)
            }
        },
        [t, report]
    )

    return (
        <button onClick={handleShare} className="share-button">
            <img src={process.env.PUBLIC_URL + '/icons/share-svgrepo-com.svg'} alt="Share" />
            <span>{t('Share.button')}</span>
        </button>
    )
}

export default ShareButton
