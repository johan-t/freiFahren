package test_getStationDistance

import (
	"fmt"
	"os"
	"reflect"
	"testing"

	"github.com/FreiFahren/backend/api/getStationDistance"
	"github.com/FreiFahren/backend/data"
)

func TestFindShortestDistance(t *testing.T) {
	os.Chdir("..")
	tests := []struct {
		name           string
		startStation   string
		userStation    string
		expectedLength int
	}{
		{"Station path should be 8 from Ernst Reuter Platz to Platz der Luftbrücke", "U-RP", "U-Pl", 8},
		{"S7 Ahrensfelde to Potsdam Hauptbahnof", "S-Ah", "S-PH", 28},
	}
	data.EmbedJSONFiles()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			distances := getStationDistance.FindShortestDistance(tt.startStation, tt.userStation)
			fmt.Printf("Distances: %v\n", distances)
			if !reflect.DeepEqual(distances, tt.expectedLength) {
				t.Errorf("FindShortestPath(%s, %s) = %v; expected %v", tt.startStation, tt.userStation, distances, tt.expectedLength)
			}
		})
	}
}
