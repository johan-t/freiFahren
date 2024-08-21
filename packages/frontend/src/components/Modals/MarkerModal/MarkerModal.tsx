import React, { useEffect, useState, useMemo, useRef } from 'react'
import { MarkerData } from '../../Map/Markers/MarkerContainer'
import { elapsedTimeMessage, stationDistanceMessage } from '../../../utils/mapUtils'
import { getStationDistance } from '../../../utils/dbUtils'
import Skeleton, { useSkeleton } from '../../Miscellaneous/LoadingPlaceholder/Skeleton'
import './MarkerModal.css'

interface MarkerModalProps {
    selectedMarker: MarkerData
    className: string
    userLat?: number
    userLng?: number
    children?: React.ReactNode
}

const MarkerModal: React.FC<MarkerModalProps> = ({ className, children, selectedMarker, userLat, userLng }) => {
    const { timestamp, station, line, direction, isHistoric } = selectedMarker

    const adjustedTimestamp = useMemo(() => {
        const tempTimestamp = new Date(timestamp)
        return tempTimestamp
    }, [timestamp])
    const currentTime = new Date().getTime()
    const elapsedTime = currentTime - adjustedTimestamp.getTime()

    const [isLoading, setIsLoading] = useState(false)
    const [stationDistance, setStationDistance] = useState<number | null>(null)
    const [shouldShowSkeleton, setShouldShowSkeleton] = useState(true)

    const prevStationId = useRef(station.id)

    const showSkeleton = useSkeleton({ isLoading: isLoading && shouldShowSkeleton })

    useEffect(() => {
        const fetchDistance = async () => {
            setIsLoading(true)
            const distance = await getStationDistance(userLat, userLng, station.id)
            setStationDistance(distance)
            setIsLoading(false)
            // to avoid showing the skeleton when pos changes due to watchPosition
            setShouldShowSkeleton(false)
        }

        // only show skeleton if the station changes
        if (station.id !== prevStationId.current) {
            setShouldShowSkeleton(true)
            setStationDistance(null)
            prevStationId.current = station.id
        }

        fetchDistance()
    }, [userLat, userLng, station.id])

    return (
        <div className={`marker-modal info-popup modal ${className}`}>
            {children}
            <h1>{station.name}</h1>
            {(direction.name !== '' || line !== '') && (
                <h2>
                    <span className={line}>{line}</span> {direction.name}
                </h2>
            )}
            <div>
                <p>{elapsedTimeMessage(elapsedTime, isHistoric)}</p>
                <p className="distance">{showSkeleton ? <Skeleton /> : stationDistanceMessage(stationDistance)}</p>
                {selectedMarker.message && <p className="description">{selectedMarker.message}</p>}
            </div>
        </div>
    )
}

export default MarkerModal
