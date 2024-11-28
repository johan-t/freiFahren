import './LegalDisclaimer.css'

import { useTranslation } from 'react-i18next'
import { Link } from 'react-router-dom'

interface LegalDisclaimerProps {
    closeModal: () => void
    openAnimationClass?: string
}

export const LegalDisclaimer = ({ closeModal, openAnimationClass }: LegalDisclaimerProps) => {
    const { t } = useTranslation()

    return (
        <div className={`legal-disclaimer container modal ${openAnimationClass}`} id="legal-disclaimer">
            <div className="content">
                <h1>{t('LegalDisclaimer.title')}</h1>
                <section>
                    <p>{t('LegalDisclaimer.text')}</p>
                    <ol>
                        <li>
                            <strong>{t('LegalDisclaimer.ticket')}</strong>
                            <p>{t('LegalDisclaimer.ticketDescription')}</p>
                        </li>
                        <li>
                            <strong>{t('LegalDisclaimer.activeUsage')}</strong>
                            <p>{t('LegalDisclaimer.activeUsageDescription')}</p>
                        </li>
                    </ol>
                    <p>{t('LegalDisclaimer.saved')}</p>
                </section>
            </div>
            <div className="footer">
                {/* eslint-disable-next-line react/button-has-type */}
                <button onClick={closeModal}>{t('LegalDisclaimer.confirm')}</button>
                <ul className="align-child-on-line">
                    <li>
                        <Link to="/impressum">{t('LegalDisclaimer.impressum')}</Link>
                    </li>
                    <li>
                        <Link to="/Datenschutz">{t('LegalDisclaimer.privacy')}</Link>
                    </li>
                </ul>
            </div>
        </div>
    )
}
