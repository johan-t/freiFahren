import React, { useState, useCallback, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { useTicketInspectors } from 'src/contexts/TicketInspectorsContext'
import { getLineColor } from 'src/utils/uiUtils'
import { sendAnalyticsEvent } from 'src/utils/analytics'
import { useLastReportView } from 'src/hooks/useLastReportView'

import './ReportsModalButton.css'

interface ReportsModalButtonProps {
    openModal: () => void
}

const ReportsModalButton: React.FC<ReportsModalButtonProps> = ({ openModal }) => {
    const { t } = useTranslation()
    const { ticketInspectorList } = useTicketInspectors()
    const [isRecent, setIsRecent] = useState(false)
    const { setLastViewed, hasViewedReport } = useLastReportView()

    const latestReport =
        ticketInspectorList.length > 0
            ? ticketInspectorList.reduce((latest, current) => {
                  const currentTime = new Date(current.timestamp).getTime()
                  const latestTime = new Date(latest.timestamp).getTime()
                  return currentTime > latestTime ? current : latest
              }, ticketInspectorList[0])
            : null

    const handleClick = () => {
        openModal()
        setLastViewed()
        sendAnalyticsEvent('ReportsModal opened', {})
    }

    const checkRecent = useCallback(() => {
        if (!latestReport) return

        const currentTime = new Date().getTime()
        const reportTime = new Date(latestReport.timestamp).getTime()
        const elapsedTime = currentTime - reportTime
        const isRecentReport = elapsedTime <= 30 * 60 * 1000 // 30 minutes

        setIsRecent(isRecentReport && !hasViewedReport(latestReport))
    }, [latestReport, hasViewedReport])

    useEffect(() => {
        checkRecent()
        const interval = setInterval(checkRecent, 30 * 1000) // Check every 30 seconds
        return () => clearInterval(interval)
    }, [checkRecent])

    return (
        <button className="list-button small-button align-child-on-line" onClick={handleClick}>
            <div className="list-button-content">
                <div className="list-button-header">
                    <p>{t('InspectorListButton.label')}</p>
                    <p>{ticketInspectorList.length}</p>
                </div>
                {latestReport && (
                    <div className="latest-report">
                        {latestReport.line && (
                            <span className="line-label" style={{ backgroundColor: getLineColor(latestReport.line) }}>
                                {latestReport.line}
                            </span>
                        )}
                        <p className="station-name">{latestReport.station.name}</p>
                        {isRecent && <span className="indicator live pulse" />}
                    </div>
                )}
            </div>
        </button>
    )
}

export default ReportsModalButton
