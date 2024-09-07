import { useState, useEffect } from 'react'

import './AskForLocation.css'
import AutocompleteInputForm from '../../Form/AutocompleteInputForm/AutocompleteInputForm'
import StationsList from '../../../data/StationsList.json'
import { useLocation } from '../../../contexts/LocationContext'

interface AskForLocationProps {
    className: string
    children?: React.ReactNode
    closeModal: () => void
}

interface Option {
    value: string
    label: string
}

const AskForLocation: React.FC<AskForLocationProps> = ({ className, children, closeModal }) => {
    const { setUserPosition } = useLocation()

    const emptyOption = ''

    const [isValid, setIsValid] = useState(false)
    const [stationInput, setStationInput] = useState(emptyOption)
    const [stationOptions, setStationOptions] = useState<Option[]>([])

    useEffect(() => {
        function populateStationOptions() {
            const options = Object.keys(StationsList).map((key) => ({
                value: key,
                label: StationsList[key as keyof typeof StationsList].name,
            }))
            setStationOptions(options)
        }

        populateStationOptions()
    }, [])

    const handleSubmit = (event: React.FormEvent) => {
        event.preventDefault()
        if (stationInput) {
            const station = StationsList[stationInput as keyof typeof StationsList]
            if (station) {
                setUserPosition({ lat: station.coordinates.latitude, lng: station.coordinates.longitude })
                closeModal()
            } else {
                console.error('Station data not found.')
            }
        } else {
            console.error('No station selected or selection is invalid.')
        }
    }

    return (
        <div className={`ask-for-location info-popup modal ${className}`}>
            {children}
            <form onSubmit={handleSubmit}>
                <h1>Was ist deine nächste Station?</h1>
                <p>Wir konnten deinen Standort nicht finden</p>

                <button type="submit" className={isValid ? '' : 'button-gray'}>
                    Standort setzen
                </button>
            </form>
        </div>
    )
}

export default AskForLocation
