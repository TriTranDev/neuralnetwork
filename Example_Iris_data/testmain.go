package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/TriTranDev/neuralnetwork"
	"gonum.org/v1/gonum/mat"
)

func loadData(name string, numberField int, nInput int, nOutput int) (*mat.Dense, *mat.Dense, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, nil, err
	}

	defer f.Close()
	reader := csv.NewReader(f)
	reader.FieldsPerRecord = numberField
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	inputsData := make([]float64, nInput*len(rawCSVData))
	labelsData := make([]float64, nOutput*len(rawCSVData))

	var inputsIndex int
	var labelsIndex int

	for idx, record := range rawCSVData {
		if idx == 0 {
			continue
		}

		for i, val := range record {
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			if i >= nInput {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}
			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}

	inputs := mat.NewDense(len(rawCSVData), nInput, inputsData)
	labels := mat.NewDense(len(rawCSVData), nOutput, labelsData)
	return inputs, labels, nil
}

func main() {
	inputNumber := 5
	outputNumber := 1
	hiddenNumber := 10
	input, labels, err := loadData("training.csv", inputNumber+outputNumber, inputNumber, outputNumber)

	testInputs, testLabels, err := loadData("test.csv", inputNumber+outputNumber, inputNumber, outputNumber)
	// predicInput, _, err := loadData("predict.csv", inputNumber+outputNumber, inputNumber, outputNumber)

	if err != nil {
		log.Fatal(err)
	}
	rowInput, _ := input.Dims()

	rowTest, _ := testInputs.Dims()

	config := neuralnetwork.NeuralNetConfig{
		InputNeurons:  inputNumber,
		OutputNeurons: outputNumber,
		HiddenNeurons: hiddenNumber,
		NumEpochs:     5000,
		LearningRate:  0.01,
	}

	inputParam := neuralnetwork.InputParamNetwork{
		Config:       config,
		CountRow:     rowInput,
		CountInput:   config.InputNeurons,
		CountOutput:  config.OutputNeurons,
		InputData:    input.RawMatrix().Data,
		OutputData:   labels.RawMatrix().Data,
		CountRowTest: rowTest,
		TestData:     testInputs.RawMatrix().Data,
		LabelData:    testLabels.RawMatrix().Data,
	}

	result := neuralnetwork.ProcessNeural(inputParam)
	fmt.Println("Correlation Coefficient ", result.CorrelationValue)

	// rowPredict, _ := predicInput.Dims()
	// fmt.Println("Predict Data ", predicInput)

	// predictData := neuralnetwork.PredictWithNeural(config, result, rowPredict, config.InputNeurons, config.OutputNeurons, predicInput.RawMatrix().Data)

	// fmt.Println("ket qua du doan la ", predictData)

}
