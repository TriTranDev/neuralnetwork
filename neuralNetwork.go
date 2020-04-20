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
	wHidden          []float64
	bHidden          []float64
	wOut             []float64
	bOut             []float64
	correlationValue float64
}

type neuralNet struct {
	config  neuralNetConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense
}

// NeuralNetConfig object
type NeuralNetConfig struct {
	inputNeurons  int
	outputNeurons int
	hiddenNeurons int
	numEpochs     int
	learningRate  float64
}

// InputParamNetwork param input for network
type InputParamNetwork struct {
	config       neuralNetConfig
	countRow     int
	countInput   int
	countOutput  int
	inputData    []float64
	outputData   []float64
	countRowTest int
	testData     []float64
	labelData    []float64
}

func newNetwork(config neuralNetConfig) *neuralNet {
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

	wHiddenRaw := make([]float64, nn.config.hiddenNeurons*nn.config.inputNeurons)
	bHiddenRaw := make([]float64, nn.config.hiddenNeurons)
	wOutRaw := make([]float64, nn.config.outputNeurons*nn.config.hiddenNeurons)
	bOutRaw := make([]float64, nn.config.outputNeurons)

	for _, param := range [][]float64{wHiddenRaw, bHiddenRaw, wOutRaw, bOutRaw} {
		for i := range param {
			param[i] = randGen.Float64()
		}
	}

	wHidden := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, wHiddenRaw)
	bHidden := mat.NewDense(1, nn.config.hiddenNeurons, bHiddenRaw)
	wOut := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, wOutRaw)
	bOut := mat.NewDense(1, nn.config.outputNeurons, bOutRaw)

	rowX, colX := x.Dims()
	rowY, colY := y.Dims()
	fmt.Printf("rowX :%d colX :%d \n", rowX, colX)
	fmt.Printf("rowY :%d colY :%d \n", rowY, colY)
	output := mat.NewDense(rowY, nn.config.outputNeurons, nil)
	for i := 0; i < nn.config.numEpochs; i++ {
		// complete the feed forward process

		hiddenLayerInput := mat.NewDense(rowX, nn.config.hiddenNeurons, nil)
		hiddenLayerInput.Mul(x, wHidden)

		addBHidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
		hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

		hiddenLayerActivations := mat.NewDense(rowX, nn.config.hiddenNeurons, nil)
		applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

		outputLayerInput := mat.NewDense(rowX, nn.config.outputNeurons, nil)
		outputLayerInput.Mul(hiddenLayerActivations, wOut)

		addBOut := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
		outputLayerInput.Apply(addBOut, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)

		//Complete the backpropagation
		networkError := mat.NewDense(rowY, nn.config.outputNeurons, nil)
		networkError.Sub(y, output)
		// calulateError := func(_, col int, v float64) float64 { return math.Pow(v, 2) / 2 }
		// networkError.Apply(calulateError, networkError)

		slopeOutputLayer := mat.NewDense(rowY, nn.config.outputNeurons, nil)
		applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }
		slopeOutputLayer.Apply(applySigmoidPrime, output)

		dOutput := mat.NewDense(rowY, nn.config.outputNeurons, nil)
		dOutput.MulElem(networkError, slopeOutputLayer)

		errorAtHiddenLayer := mat.NewDense(rowY, nn.config.hiddenNeurons, nil)
		errorAtHiddenLayer.Mul(dOutput, wOut.T())

		slopeHiddenLayer := mat.NewDense(rowY, nn.config.hiddenNeurons, nil)
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

		dHiddenLayer := mat.NewDense(rowY, nn.config.hiddenNeurons, nil)
		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

		//Adjust the parameters
		wOutAdj := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, nil)
		wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)

		wOutAdj.Scale(nn.config.learningRate, wOutAdj)

		wOut.Add(wOut, wOutAdj)

		bOutAdj, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		bOutAdj.Scale(nn.config.learningRate, bOutAdj)
		bOut.Add(bOut, bOutAdj)

		wHiddenAdj := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, nil)
		wHiddenAdj.Mul(x.T(), dHiddenLayer)
		wHiddenAdj.Scale(nn.config.learningRate, wHiddenAdj)
		wHidden.Add(wHidden, wHiddenAdj)

		bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
		if err != nil {
			return err
		}
		bHiddenAdj.Scale(nn.config.learningRate, bHiddenAdj)
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
	outPut := mat.NewDense(rowX, nn.config.outputNeurons, nil)
	// outPut := mat.NewDense(0, 0, nil)

	hiddenLayerInput := mat.NewDense(rowX, nn.config.hiddenNeurons, nil)
	hiddenLayerInput.Mul(x, nn.wHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + nn.bHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := mat.NewDense(rowX, nn.config.hiddenNeurons, nil)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := mat.NewDense(rowX, nn.config.outputNeurons, nil)
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
	network := newNetwork(param.config)
	input := mat.NewDense(param.countRow, param.countInput, param.inputData)
	label := mat.NewDense(param.countRow, param.countOutput, param.outputData)
	testInputs := mat.NewDense(param.countRowTest, param.countInput, param.testData)
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

		result = correlationCoefficient(rawPredict, param.labelData, numPreds)
	}

	wHidden := network.wHidden.RawMatrix().Data
	bHidden := network.bHidden.RawMatrix().Data
	wOut := network.wOut.RawMatrix().Data
	bOut := network.bOut.RawMatrix().Data

	resultNetwork := ResultNetWork{
		wHidden:          wHidden,
		bHidden:          bHidden,
		wOut:             wOut,
		bOut:             bOut,
		correlationValue: result,
	}
	return resultNetwork
}

//PredictWithNeural predict data
func PredictWithNeural(config neuralNetConfig, dataNetWork ResultNetWork, rows int, countInput int, countOutput int, inputData []float64) []float64 {
	network := newNetwork(config)
	wHidden := mat.NewDense(config.inputNeurons, config.hiddenNeurons, dataNetWork.wHidden)
	bHidden := mat.NewDense(1, config.hiddenNeurons, dataNetWork.bHidden)
	wOut := mat.NewDense(config.hiddenNeurons, config.outputNeurons, dataNetWork.wOut)
	bOut := mat.NewDense(1, config.outputNeurons, dataNetWork.bOut)
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
