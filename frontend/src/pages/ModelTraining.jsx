import React, { useState } from 'react';
import {
  Box,
  Heading,
  Text,
  Button,
  FormControl,
  FormLabel,
  Select,
  SimpleGrid,
  Card,
  CardHeader,
  CardBody,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  useToast,
  Spinner,
  Alert,
  AlertIcon,
  Switch,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Badge,
  IconButton,
  Flex,
  Progress,
} from '@chakra-ui/react';
import { DeleteIcon } from '@chakra-ui/icons';

// Mock data
const mockModelInfo = [
  {
    symbol: 'BTC',
    model_type: 'lstm',
    training_date: '2025-04-01T10:00:00Z',
    metrics: {
      mse: 0.0025,
      mae: 0.0345,
      r2: 0.87
    }
  },
  {
    symbol: 'ETH',
    model_type: 'xgboost',
    training_date: '2025-04-02T14:30:00Z',
    metrics: {
      mse: 0.0018,
      mae: 0.0289,
      r2: 0.91
    }
  },
  {
    symbol: 'SOL',
    model_type: 'gru',
    training_date: '2025-04-03T09:15:00Z',
    metrics: {
      mse: 0.0031,
      mae: 0.0412,
      r2: 0.83
    }
  }
];

const ModelTraining = () => {
  const [selectedCrypto, setSelectedCrypto] = useState('BTC');
  const [selectedModel, setSelectedModel] = useState('lstm');
  const [epochs, setEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(32);
  const [lookback, setLookback] = useState(30);
  const [horizon, setHorizon] = useState(7);
  const [useSentiment, setUseSentiment] = useState(true);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const toast = useToast();

  // Handle train model
  const handleTrainModel = () => {
    setIsTraining(true);
    setTrainingProgress(0);
    
    // Simulate training progress
    const interval = setInterval(() => {
      setTrainingProgress(prev => {
        const newProgress = prev + Math.random() * 10;
        if (newProgress >= 100) {
          clearInterval(interval);
          setIsTraining(false);
          toast({
            title: 'Model trained successfully',
            description: `Trained ${selectedModel.toUpperCase()} model for ${selectedCrypto} with ${epochs} epochs`,
            status: 'success',
            duration: 5000,
            isClosable: true,
          });
          return 100;
        }
        return newProgress;
      });
    }, 500);
  };

  // Handle delete model
  const handleDeleteModel = (symbol, modelType) => {
    toast({
      title: 'Model deleted',
      description: `Deleted ${modelType.toUpperCase()} model for ${symbol}`,
      status: 'info',
      duration: 5000,
      isClosable: true,
    });
  };

  return (
    <Box p={5}>
      <Heading mb={5}>Model Training</Heading>
      
      <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={5}>
        {/* Train Model Card */}
        <Card>
          <CardHeader>
            <Heading size="md">Train New Model</Heading>
          </CardHeader>
          <CardBody>
            <SimpleGrid columns={{ base: 1, md: 2 }} spacing={5}>
              <FormControl>
                <FormLabel>Cryptocurrency</FormLabel>
                <Select 
                  value={selectedCrypto}
                  onChange={(e) => setSelectedCrypto(e.target.value)}
                >
                  <option value="BTC">Bitcoin (BTC)</option>
                  <option value="ETH">Ethereum (ETH)</option>
                  <option value="SOL">Solana (SOL)</option>
                  <option value="BNB">Binance Coin (BNB)</option>
                  <option value="XRP">Ripple (XRP)</option>
                </Select>
              </FormControl>
              
              <FormControl>
                <FormLabel>Model Type</FormLabel>
                <Select 
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                >
                  <option value="lstm">LSTM</option>
                  <option value="gru">GRU</option>
                  <option value="xgboost">XGBoost</option>
                  <option value="lightgbm">LightGBM</option>
                </Select>
              </FormControl>
              
              <FormControl>
                <FormLabel>Epochs</FormLabel>
                <NumberInput 
                  value={epochs} 
                  onChange={(valueString) => setEpochs(parseInt(valueString))}
                  min={1}
                  max={1000}
                >
                  <NumberInputField />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
              </FormControl>
              
              <FormControl>
                <FormLabel>Batch Size</FormLabel>
                <NumberInput 
                  value={batchSize} 
                  onChange={(valueString) => setBatchSize(parseInt(valueString))}
                  min={1}
                  max={256}
                >
                  <NumberInputField />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
              </FormControl>
              
              <FormControl>
                <FormLabel>Lookback Days</FormLabel>
                <NumberInput 
                  value={lookback} 
                  onChange={(valueString) => setLookback(parseInt(valueString))}
                  min={1}
                  max={365}
                >
                  <NumberInputField />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
              </FormControl>
              
              <FormControl>
                <FormLabel>Prediction Horizon</FormLabel>
                <NumberInput 
                  value={horizon} 
                  onChange={(valueString) => setHorizon(parseInt(valueString))}
                  min={1}
                  max={30}
                >
                  <NumberInputField />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
              </FormControl>
              
              <FormControl display="flex" alignItems="center">
                <FormLabel mb="0">
                  Include Sentiment Analysis
                </FormLabel>
                <Switch 
                  isChecked={useSentiment}
                  onChange={() => setUseSentiment(!useSentiment)}
                />
              </FormControl>
            </SimpleGrid>
            
            {isTraining && (
              <Box mt={5}>
                <Text mb={2}>Training Progress: {Math.round(trainingProgress)}%</Text>
                <Progress value={trainingProgress} size="sm" colorScheme="blue" />
              </Box>
            )}
            
            <Button 
              mt={5} 
              colorScheme="blue" 
              onClick={handleTrainModel}
              isLoading={isTraining}
              loadingText="Training..."
              isDisabled={isTraining}
            >
              Train Model
            </Button>
          </CardBody>
        </Card>
        
        {/* Trained Models Card */}
        <Card>
          <CardHeader>
            <Heading size="md">Trained Models</Heading>
          </CardHeader>
          <CardBody>
            {isTraining ? (
              <Box textAlign="center" py={10}>
                <Spinner size="xl" />
                <Text mt={3}>Loading models...</Text>
              </Box>
            ) : mockModelInfo.length > 0 ? (
              <Table variant="simple">
                <Thead>
                  <Tr>
                    <Th>Symbol</Th>
                    <Th>Model</Th>
                    <Th>Training Date</Th>
                    <Th>Metrics</Th>
                    <Th>Actions</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  {mockModelInfo.map((model, index) => (
                    <Tr key={index}>
                      <Td>{model.symbol}</Td>
                      <Td>
                        <Badge colorScheme={
                          model.model_type === 'lstm' ? 'blue' :
                          model.model_type === 'gru' ? 'purple' :
                          model.model_type === 'xgboost' ? 'green' :
                          'orange'
                        }>
                          {model.model_type.toUpperCase()}
                        </Badge>
                      </Td>
                      <Td>{new Date(model.training_date).toLocaleString()}</Td>
                      <Td>
                        <Text fontSize="sm">MSE: {model.metrics.mse.toFixed(4)}</Text>
                        <Text fontSize="sm">MAE: {model.metrics.mae.toFixed(4)}</Text>
                        <Text fontSize="sm">R²: {model.metrics.r2.toFixed(2)}</Text>
                      </Td>
                      <Td>
                        <IconButton
                          aria-label="Delete model"
                          icon={<DeleteIcon />}
                          colorScheme="red"
                          variant="ghost"
                          onClick={() => handleDeleteModel(model.symbol, model.model_type)}
                        />
                      </Td>
                    </Tr>
                  ))}
                </Tbody>
              </Table>
            ) : (
              <Alert status="info">
                <AlertIcon />
                No trained models available. Train a model first.
              </Alert>
            )}
          </CardBody>
        </Card>
      </SimpleGrid>
    </Box>
  );
};

export default ModelTraining;
