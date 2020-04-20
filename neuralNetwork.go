package neuralnetwork

import (
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	return x * (1.0 - x)
}

// ResultNetWork save result of network
type ResultNetWork struct {
	WHidden          []float64
	BHidden          []float64
	WOut             []float64
	BOut             []float64
	CorrelationValue float64
}

// NeuralNetConfig object
type NeuralNetConfig struct {
	InputNeurons  int
	OutputNeurons int
	HiddenNeurons int
	NumEpochs     int
	LearningRate  float64
}

type neuralNet struct {
	config  NeuralNetConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense
}

// InputParamNetwork param input for network
type InputParamNetwork struct {
	Config       NeuralNetConfig
	CountRow     int
	CountInput   int
	CountOutput  int
	InputData    []float64
	OutputData   []float64
	CountRowTest int
	TestData     []float64
	LabelData    []float64
}

func newNetwork(config NeuralNetConfig) *neuralNet {
	return &neuralNet{config: config}
}

func printMatrix(x *mat.Dense, name string) {
	f := mat.Formatted(x, mat.Prefix(" "))
	fmt.Println(name)
	fmt.Printf("%v\n", f)
	fmt.Println()
}

func (nn *neuralNet) train(x, y *mat.Dense) error {
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	wHiddenRaw := make([]float64, nn.config.HiddenNeurons*nn.config.InputNeurons)
	bHiddenRaw := make([]float64, nn.config.HiddenNeurons)
	wOutRaw := make([]float64, nn.config.OutputNeurons*nn.config.HiddenNeurons)
	bOutRaw := make([]float64, nn.config.OutputNeurons)

	for _, param := range [][]float64{wHiddenRaw, bHiddenRaw, wOutRaw, bOutRaw} {
		for i := range param {
			param[i] = randGen.Float64()
		}
	}

	wHidden := mat.NewDense(nn.config.InputNeurons, nn.config.HiddenNeurons, wHiddenRaw)
	bHidden := mat.NewDense(1, nn.config.HiddenNeurons, bHiddenRaw)
	wOut := mat.NewDense(nn.config.HiddenNeurons, nn.config.OutputNeurons, wOutRaw)
	bOut := mat.NewDense(1, nn.config.OutputNeurons, bOutRaw)

	rowX, colX := x.Dims()
	rowY, colY := y.Dims()
	fmt.Printf("rowX :%d colX :%d \n", rowX, colX)
	fmt.Printf("rowY :%d colY :%d \n", rowY, colY)
	output := mat.NewDense(rowY, nn.config.OutputNeurons, nil)
	for i := 0; i < nn.config.NumEpochs; i++ {
		// complete the feed forward process

		hiddenLayerInput := mat.NewDense(rowX, nn.config.HiddenNeurons, nil)
		hiddenLayerInput.Mul(x, wHidden)

		addBHidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
		hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

		hiddenLayerActivations := mat.NewDense(rowX, nn.config.HiddenNeurons, nil)
		applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

		outputLayerInput := mat.NewDense(rowX, nn.config.OutputNeurons, nil)
		outputLayerInput.Mul(hiddenLayerActivations, wOut)

		addBOut := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
		outputLayerInput.Apply(addBOut, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)

		//Complete the backpropagation
		networkError := mat.NewDense(rowY, nn.config.OutputNeurons, nil)
		networkError.Sub(y, output)
		// calulateError := func(_, col int, v float64) float64 { return math.Pow(v, 2) / 2 }
		// networkError.Apply(calulateError, networkError)

		slopeOutputLayer := mat.NewDense(rowY, nn.config.OutputNeurons, nil)
		applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }
		slopeOutputLayer.Apply(applySigmoidPrime, output)

		dOutput := mat.NewDense(rowY, nn.config.OutputNeurons, nil)
		dOutput.MulElem(networkError, slopeOutputLayer)

		errorAtHiddenLayer := mat.NewDense(rowY, nn.config.HiddenNeurons, nil)
		errorAtHiddenLayer.Mul(dOutput, wOut.T())

		slopeHiddenLayer := mat.NewDense(rowY, nn.config.HiddenNeurons, nil)
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

		dHiddenLayer := mat.NewDense(rowY, nn.config.HiddenNeurons, nil)
		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

		//Adjust the parameters
		wOutAdj := mat.NewDense(nn.config.HiddenNeurons, nn.config.OutputNeurons, nil)
		wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)

		wOutAdj.Scale(nn.config.LearningRate, wOutAdj)

		wOut.Add(wOut, wOutAdj)

		bOutAdj, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		bOutAdj.Scale(nn.config.LearningRate, bOutAdj)
		bOut.Add(bOut, bOutAdj)

		wHiddenAdj := mat.NewDense(nn.config.InputNeurons, nn.config.HiddenNeurons, nil)
		wHiddenAdj.Mul(x.T(), dHiddenLayer)
		wHiddenAdj.Scale(nn.config.LearningRate, wHiddenAdj)
		wHidden.Add(wHidden, wHiddenAdj)

		bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
		if err != nil {
			return err
		}
		bHiddenAdj.Scale(nn.config.LearningRate, bHiddenAdj)
		bHidden.Add(bHidden, bHiddenAdj)
	}

	nn.wHidden = wHidden
	nn.bHidden = bHidden
	nn.wOut = wOut
	nn.bOut = bOut

	return nil
}

//sumAlongAxis sums a matrix along a
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {
	numRows, numCols := m.Dims()
	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("Invalid axis, must be 0 or 1")
	}

	return output, nil
}

func (nn *neuralNet) predict(x *mat.Dense) (*mat.Dense, error) {
	// check  to make sure that out neural net value
	// represents a trained model
	if nn.wHidden == nil || nn.wOut == nil || nn.bHidden == nil || nn.bOut == nil {
		return nil, errors.New("The supplied neural net weight and biases are empty")
	}

	//Define the output of the neural network
	rowX, colX := x.Dims()
	// rowY, colY := y.Dims()
	fmt.Printf("rowX :%d colX :%d \n", rowX, colX)
	// fmt.Printf("rowY :%d colY :%d \n", rowY, colY)
	outPut := mat.NewDense(rowX, nn.config.OutputNeurons, nil)
	// outPut := mat.NewDense(0, 0, nil)

	hiddenLayerInput := mat.NewDense(rowX, nn.config.HiddenNeurons, nil)
	hiddenLayerInput.Mul(x, nn.wHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + nn.bHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := mat.NewDense(rowX, nn.config.HiddenNeurons, nil)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := mat.NewDense(rowX, nn.config.OutputNeurons, nil)
	outputLayerInput.Mul(hiddenLayerActivations, nn.wOut)
	addBOut := func(_, col int, v float64) float64 { return v + nn.bOut.At(0, col) }
	outputLayerInput.Apply(addBOut, outputLayerInput)
	outPut.Apply(applySigmoid, outputLayerInput)

	return outPut, nil

}

func correlationCoefficient(X []float64, Y []float64, n int) float64 {

	sumX := 0.0
	sumY := 0.0
	sumXY := 0.0
	squareSumX := 0.0
	squareSumY := 0.0

	for i := 0; i < n; i++ {
		// sum of elements of array X.
		sumX = sumX + X[i]

		// sum of elements of array Y.
		sumY = sumY + Y[i]

		// sum of X[i] * Y[i].
		sumXY = sumXY + X[i]*Y[i]

		// sum of square of array elements.
		squareSumX = squareSumX + X[i]*X[i]
		squareSumY = squareSumY + Y[i]*Y[i]
	}

	// use formula for calculating correlation
	// coefficient.
	nn := float64(n)
	corr := float64((nn*sumXY - sumX*sumY)) /
		(math.Sqrt(float64((nn*squareSumX - sumX*sumX) * (nn*squareSumY - sumY*sumY))))

	return corr

}

//ProcessNeural main function
func ProcessNeural(param InputParamNetwork) ResultNetWork {
	network := newNetwork(param.Config)
	input := mat.NewDense(param.CountRow, param.CountInput, param.InputData)
	label := mat.NewDense(param.CountRow, param.CountOutput, param.OutputData)
	testInputs := mat.NewDense(param.CountRowTest, param.CountInput, param.TestData)
	// testLabel := mat.NewDense(param.countRowTest, param.countOutput, param.labelData)

	var result float64
	var countLoop int

	for result <= 0.98 && countLoop <= 10 {
		countLoop++
		if err := network.train(input, label); err != nil {
			log.Fatal(err)
		}

		predictions, err := network.predict(testInputs)
		if err != nil {
			log.Fatal(err)
		}

		numPreds, _ := predictions.Dims()

		rawPredict := predictions.RawMatrix().Data

		result = correlationCoefficient(rawPredict, param.LabelData, numPreds)
	}

	wHidden := network.wHidden.RawMatrix().Data
	bHidden := network.bHidden.RawMatrix().Data
	wOut := network.wOut.RawMatrix().Data
	bOut := network.bOut.RawMatrix().Data

	resultNetwork := ResultNetWork{
		WHidden:          wHidden,
		BHidden:          bHidden,
		WOut:             wOut,
		BOut:             bOut,
		CorrelationValue: result,
	}
	return resultNetwork
}

//PredictWithNeural predict data
func PredictWithNeural(config NeuralNetConfig, dataNetWork ResultNetWork, rows int, countInput int, countOutput int, inputData []float64) []float64 {
	network := newNetwork(config)
	wHidden := mat.NewDense(config.InputNeurons, config.HiddenNeurons, dataNetWork.WHidden)
	bHidden := mat.NewDense(1, config.HiddenNeurons, dataNetWork.BHidden)
	wOut := mat.NewDense(config.HiddenNeurons, config.OutputNeurons, dataNetWork.WOut)
	bOut := mat.NewDense(1, config.OutputNeurons, dataNetWork.BOut)
	inputLabel := mat.NewDense(rows, countInput, inputData)

	network.wHidden = wHidden
	network.bHidden = bHidden
	network.wOut = wOut
	network.bOut = bOut

	predict, err := network.predict(inputLabel)
	if err != nil {
		log.Fatal(err)
	}

	return predict.RawMatrix().Data
}
