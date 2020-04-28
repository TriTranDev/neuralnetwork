# neuralnetwork

## What is it?
This is back-propagation  neural network for only one hidden layer. I will update it for muilti hidden layer later.

## Installation
Use go get to install this package

 ```bash
 go get github.com/TriTranDev/neuralnetwork
 ```

## Usage

Please check Example.<br />
first create config <br />
```golang
config := neuralnetwork.NeuralNetConfig{
		InputNeurons:  inputNumber,
		OutputNeurons: outputNumber,
		HiddenNeurons: hiddenNumber,
		NumEpochs:     5000,
		LearningRate:  0.01,
	}
```

inputNumber: lenght of your input put data <br />
outputNumber: lenght of your out put data <br />
hiddenNumber: lenght of your hidden you want to use example 10,20,100.... <br />
NumEpochs: number of loops when you train the neural <br />
LearningRate: it use for training neural, check google for more information. <br />

```golang
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
```

this is param for build the network <br />

```golang
result := neuralnetwork.ProcessNeural(inputParam)
	fmt.Println("Correlation Coefficient ", result.CorrelationValue)
```

call ProcessNeural to build the network and check the [Correlation Coefficient](https://www.investopedia.com/terms/c/correlationcoefficient.asp)

## Note
Maybe prepare your data if you use different data format

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
