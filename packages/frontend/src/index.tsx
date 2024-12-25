import './index.css'

import React from 'react'
import ReactDOM from 'react-dom/client'
import { I18nextProvider } from 'react-i18next'
import { BrowserRouter, Route, Routes } from 'react-router-dom'

import { LocationProvider } from './contexts/LocationContext'
import { i18n } from './i18n'
import { App } from './pages/App/App'
import { PrivacyPolicy } from './pages/PrivacyPolicy/PrivacyPolicy'
import { Support } from './pages/Support/Support'
import { reportWebVitals } from './reportWebVitals'

export type FunnelConfig = {
    path: string
    source: string
}

const FUNNEL_ROUTES: FunnelConfig[] = [
    {
        path: '/invite',
        source: 'FreiFahren_BE Telegram',
    },
]

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement)

root.render(
    <React.StrictMode>
        <I18nextProvider i18n={i18n}>
            <BrowserRouter>
                <Routes>
                    <Route
                        path="/"
                        element={
                            <LocationProvider>
                                <App />
                            </LocationProvider>
                        }
                    />
                    <Route path="/datenschutz" element={<PrivacyPolicy />} />
                    <Route path="/support" element={<Support />} />
                    {FUNNEL_ROUTES.map((config) => (
                        <Route
                            key={config.path}
                            path={config.path}
                            element={
                                <LocationProvider>
                                    <App funnelEvent={config} />
                                </LocationProvider>
                            }
                        />
                    ))}
                </Routes>
            </BrowserRouter>
        </I18nextProvider>
    </React.StrictMode>
)

reportWebVitals()
