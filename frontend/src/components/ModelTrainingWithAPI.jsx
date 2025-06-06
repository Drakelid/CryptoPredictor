import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
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
  Progress,
  Flex,
  Badge,
  Switch,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  IconButton,
  useColorModeValue,
} from '@chakra-ui/react';
import { DeleteIcon } from '@chakra-ui/icons';
import { keyframes } from '@emotion/react';
import { useQuery, useMutation } from 'react-query';

// Import configuration
import { API_BASE_URL, AVAILABLE_CRYPTOCURRENCIES, AVAILABLE_MODEL_TYPES } from '../config';

// Define animations
const pulseAnimation = keyframes`
  0% { box-shadow: 0 0 0 0 rgba(66, 153, 225, 0.4); }
  70% { box-shadow: 0 0 0 10px rgba(66, 153, 225, 0); }
  100% { box-shadow: 0 0 0 0 rgba(66, 153, 225, 0); }
`;


const ModelTrainingWithAPI = () => {
  const navigate = useNavigate();
  const toast = useToast();

  // Form state
  const [selectedCrypto, setSelectedCrypto] = useState('BTC');
  const [selectedModel, setSelectedModel] = useState('lstm');
  const [epochs, setEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(32);
  const [lookback, setLookback] = useState(30);
  const [horizon, setHorizon] = useState(7);
  const [useSentiment, setUseSentiment] = useState(true);
  const [hyperparameterTuning, setHyperparameterTuning] = useState(false);

  // Training status
  const [trainingStatus, setTrainingStatus] = useState('idle'); // idle, training, success, error
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingMessage, setTrainingMessage] = useState('');

  // Fetch model info
  const { data: modelInfo, isLoading: modelInfoLoading, error: modelInfoError, refetch: refetchModelInfo } = useQuery(
    ['modelInfo'],
    async () => {
      try {
        console.log('Fetching model info');
        const axiosInstance = axios.create({
          baseURL: API_BASE_URL
        });
        const response = await axiosInstance.get('/api/models/info');
        console.log('Model info response:', response.data);
        return response.data;
      } catch (error) {
        console.error('Error fetching model info:', error);
        throw error;
      }
    },
    {
      onError: (error) => {
        toast({
          title: 'Error fetching model info',
          description: error.message,
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      },
      select: (data) => data
    }
  );

  // Store progress interval reference
  const progressIntervalRef = useRef(null);

  // Cleanup function to ensure interval is cleared
  useEffect(() => {
    // Cleanup function that runs when component unmounts
    return () => {
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
      }
    };
  }, []);

  // Train model mutation
  const trainModelMutation = useMutation(
    async (params) => {
      try {
        // Clear any existing interval first
        if (progressIntervalRef.current) {
          clearInterval(progressIntervalRef.current);
          progressIntervalRef.current = null;
        }

        // Reset progress and set status to training
        setTrainingProgress(0);
        setTrainingStatus('training');
        setTrainingMessage(`Training ${params.model_type.toUpperCase()} model for ${params.symbol}...`);

        // Simulate progress updates with a safer implementation
        progressIntervalRef.current = setInterval(() => {
          setTrainingProgress(prev => {
            const newProgress = prev + Math.random() * 5;
            return newProgress >= 90 ? 90 : newProgress; // Cap at 90% until complete
          });
        }, 1000);

        // Make the actual API call
        console.log('Training model with params:', params);
        const axiosInstance = axios.create({
          baseURL: API_BASE_URL,
          timeout: 300000, // 5 minute timeout for long training operations
        });
        const response = await axiosInstance.post('/api/models/train', params);

        // Clear the interval and set progress to 100%
        if (progressIntervalRef.current) {
          clearInterval(progressIntervalRef.current);
          progressIntervalRef.current = null;
        }

        setTrainingProgress(100);
        setTrainingStatus('success');
        setTrainingMessage(`Successfully trained ${params.model_type.toUpperCase()} model for ${params.symbol}!`);

        return response.data;
      } catch (error) {
        console.error('Error training model:', error);

        // Clear the interval to prevent memory leaks
        if (progressIntervalRef.current) {
          clearInterval(progressIntervalRef.current);
          progressIntervalRef.current = null;
        }

        // Set status to error
        setTrainingStatus('error');
        setTrainingMessage(`Error: ${error.response?.data?.detail || error.message}`);
        setTrainingProgress(0);

        // Throw the error to be handled by onError
        throw error;
      }
    },
    {
      onSuccess: (data) => {
        try {
          console.log('Train model success:', data);

          // Ensure interval is cleared here too
          if (progressIntervalRef.current) {
            clearInterval(progressIntervalRef.current);
            progressIntervalRef.current = null;
          }

          // Show success message
          toast({
            title: 'Model trained successfully',
            description: `Trained ${data.model_type.toUpperCase()} model for ${data.symbol}`,
            status: 'success',
            duration: 5000,
            isClosable: true,
          });

          // Set success state immediately
          setTrainingStatus('success');
          setTrainingMessage(`Successfully trained ${data.model_type.toUpperCase()} model for ${data.symbol}!`);
          setTrainingProgress(100);

          // Refresh the model info safely
          console.log('Refreshing model info...');
          try {
            refetchModelInfo();
          } catch (refetchError) {
            console.error('Error refetching model info:', refetchError);
          }

          // Reset status after a delay (using a single setTimeout)
          setTimeout(() => {
            try {
              setTrainingStatus('idle');
              setTrainingMessage('');
              console.log('Reset training status to idle');
            } catch (resetError) {
              console.error('Error resetting training status:', resetError);
            }
          }, 4000);
        } catch (successError) {
          console.error('Error in success handler:', successError);
          // Ensure we still reset the state
          setTrainingStatus('idle');
          setTrainingMessage('');
        }
      },
      onError: (error) => {
        console.error('Train model error:', error);

        // Ensure interval is cleared here too as a safety measure
        if (progressIntervalRef.current) {
          clearInterval(progressIntervalRef.current);
          progressIntervalRef.current = null;
        }

        // Show a more detailed error message
        let errorMessage = 'An unknown error occurred';

        if (error.response?.data?.detail) {
          errorMessage = error.response.data.detail;
        } else if (error.message) {
          errorMessage = error.message;
        }

        // Log additional details for debugging
        if (error.response) {
          console.error('Error response:', error.response);
        }

        toast({
          title: 'Error training model',
          description: errorMessage,
          status: 'error',
          duration: 7000,
          isClosable: true,
        });

        // Reset status after a delay
        setTimeout(() => {
          setTrainingStatus('idle');
          setTrainingMessage('');
        }, 3000);
      }
    }
  );

  // Delete model mutation
  const deleteModelMutation = useMutation(
    async ({ symbol, model_type }) => {
      try {
        console.log(`Deleting ${model_type} model for ${symbol}`);
        const axiosInstance = axios.create({
          baseURL: API_BASE_URL
        });
        const response = await axiosInstance.delete(`/api/models/${symbol}/${model_type}`);
        return response.data;
      } catch (error) {
        console.error('Error deleting model:', error);
        throw error;
      }
    },
    {
      onSuccess: (data, variables) => {
        toast({
          title: 'Model deleted',
          description: `Deleted ${variables.model_type.toUpperCase()} model for ${variables.symbol}`,
          status: 'success',
          duration: 5000,
          isClosable: true,
        });
        // Refresh the model info
        refetchModelInfo();
      },
      onError: (error) => {
        toast({
          title: 'Error deleting model',
          description: error.response?.data?.detail || error.message,
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      }
    }
  );

  // Handle train model
  const handleTrainModel = () => {
    try {
      // Validate inputs
      if (!selectedCrypto) {
        toast({
          title: 'Please select a cryptocurrency',
          status: 'warning',
          duration: 3000,
          isClosable: true,
        });
        return;
      }

      if (!selectedModel) {
        toast({
          title: 'Please select a model type',
          status: 'warning',
          duration: 3000,
          isClosable: true,
        });
        return;
      }

      // Clear any existing interval first
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
      }

      // Reset any previous state
      setTrainingProgress(0);

      // Prepare training parameters
      const params = {
        symbol: selectedCrypto,
        model_type: selectedModel,
        lookback: lookback,
        horizon: horizon,
        epochs: epochs,
        batch_size: batchSize,
        hyperparameter_tuning: hyperparameterTuning
      };

      console.log('Starting model training with params:', params);

      // Call the mutation
      trainModelMutation.mutate(params);
    } catch (error) {
      console.error('Error in handleTrainModel:', error);

      // Clear any interval
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
      }

      // Show error message
      toast({
        title: 'Error',
        description: `An unexpected error occurred: ${error.message}`,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });

      // Reset state
      setTrainingStatus('idle');
      setTrainingProgress(0);
      setTrainingMessage('');
    }
  };

  // Handle delete model
  const handleDeleteModel = (symbol, model_type) => {
    try {
      console.log(`Deleting model: ${model_type} for ${symbol}`);
      deleteModelMutation.mutate({ symbol, model_type });
    } catch (error) {
      console.error('Error in handleDeleteModel:', error);
      toast({
        title: 'Error',
        description: `Failed to delete model: ${error.message}`,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  // Handle go to data
  const handleGoToData = () => {
    navigate('/data');
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
            {/* Info box */}
            <Alert status="info" mb={4}>
              <AlertIcon />
              <Box>
                <Text>Before training a model, make sure you have fetched or uploaded cryptocurrency data.</Text>
                <Button size="sm" colorScheme="blue" mt={2} onClick={handleGoToData}>
                  Go to Data Management
                </Button>
              </Box>
            </Alert>

            {/* Status animation */}
            {trainingStatus !== 'idle' && (
              <Box mb={6} p={4} borderWidth="1px" borderRadius="lg"
                   bg={useColorModeValue(
                     trainingStatus === 'training' ? 'blue.50' :
                     trainingStatus === 'success' ? 'green.50' : 'red.50',
                     trainingStatus === 'training' ? 'blue.900' :
                     trainingStatus === 'success' ? 'green.900' : 'red.900'
                   )}
                   animation={trainingStatus === 'training' ? `${pulseAnimation} 2s infinite` : 'none'}
                   transition="all 0.3s ease"
              >
                <Flex align="center" mb={2}>
                  {trainingStatus === 'training' && (
                    <Spinner size="sm" color="blue.500" mr={2} />
                  )}
                  <Badge colorScheme={
                    trainingStatus === 'training' ? 'blue' :
                    trainingStatus === 'success' ? 'green' : 'red'
                  }>
                    {trainingStatus === 'training' ? 'Training Model' :
                     trainingStatus === 'success' ? 'Success' : 'Error'}
                  </Badge>
                </Flex>
                <Text fontSize="sm" mb={2}>{trainingMessage}</Text>
                {trainingStatus === 'training' && (
                  <Progress
                    value={trainingProgress}
                    size="sm"
                    colorScheme="blue"
                    hasStripe
                    isAnimated
                    borderRadius="full"
                  />
                )}
              </Box>
            )}

            <SimpleGrid columns={{ base: 1, md: 2 }} spacing={5}>
              <FormControl>
                <FormLabel>Cryptocurrency</FormLabel>
                <Select
                  value={selectedCrypto}
                  onChange={(e) => setSelectedCrypto(e.target.value)}
                  isDisabled={trainingStatus === 'training'}
                >
                  {AVAILABLE_CRYPTOCURRENCIES.map(crypto => (
                    <option key={crypto.value} value={crypto.value}>{crypto.label}</option>
                  ))}
                </Select>
              </FormControl>

              <FormControl>
                <FormLabel>Model Type</FormLabel>
                <Select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  isDisabled={trainingStatus === 'training'}
                >
                  {AVAILABLE_MODEL_TYPES.map(model => (
                    <option key={model.value} value={model.value}>{model.label}</option>
                  ))}
                </Select>
              </FormControl>

              <FormControl>
                <FormLabel>Epochs</FormLabel>
                <NumberInput
                  value={epochs}
                  onChange={(valueString) => setEpochs(parseInt(valueString))}
                  min={1}
                  max={1000}
                  isDisabled={trainingStatus === 'training'}
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
                  isDisabled={trainingStatus === 'training'}
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
                  isDisabled={trainingStatus === 'training'}
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
                  isDisabled={trainingStatus === 'training'}
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
                  isDisabled={trainingStatus === 'training'}
                />
              </FormControl>

              <FormControl display="flex" alignItems="center">
                <FormLabel mb="0">
                  Hyperparameter Tuning
                </FormLabel>
                <Switch
                  isChecked={hyperparameterTuning}
                  onChange={() => setHyperparameterTuning(!hyperparameterTuning)}
                  isDisabled={trainingStatus === 'training'}
                />
              </FormControl>
            </SimpleGrid>

            <Button
              mt={5}
              colorScheme="blue"
              onClick={handleTrainModel}
              isLoading={trainingStatus === 'training'}
              loadingText="Training..."
              isDisabled={trainingStatus === 'training'}
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
            {modelInfoLoading ? (
              <Box textAlign="center" py={10}>
                <Spinner size="xl" />
                <Text mt={3}>Loading models...</Text>
              </Box>
            ) : modelInfo && modelInfo.length > 0 ? (
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
                  {modelInfo.map((model, index) => (
                    <Tr key={index}>
                      <Td>{model.symbol}</Td>
                      <Td>
                        <Badge colorScheme={
                          !model.model_type ? 'gray' :
                          model.model_type === 'lstm' ? 'blue' :
                          model.model_type === 'gru' ? 'purple' :
                          model.model_type === 'xgboost' ? 'green' :
                          'orange'
                        }>
                          {model.model_type ? model.model_type.toUpperCase() : 'UNKNOWN'}
                        </Badge>
                      </Td>
                      <Td>
                        {model.training_date ? (
                          (() => {
                            try {
                              return new Date(model.training_date).toLocaleString();
                            } catch (error) {
                              console.error('Error formatting date:', error);
                              return 'Invalid date';
                            }
                          })()
                        ) : 'Not available'}
                      </Td>
                      <Td>
                        {model.metrics ? (
                          <>
                            <Text fontSize="sm">MSE: {model.metrics.mse !== undefined ? model.metrics.mse.toFixed(4) : 'N/A'}</Text>
                            <Text fontSize="sm">MAE: {model.metrics.mae !== undefined ? model.metrics.mae.toFixed(4) : 'N/A'}</Text>
                            <Text fontSize="sm">R²: {model.metrics.r2 !== undefined ? model.metrics.r2.toFixed(2) : 'N/A'}</Text>
                          </>
                        ) : (
                          <Text fontSize="sm">Metrics not available</Text>
                        )}
                      </Td>
                      <Td>
                        <IconButton
                          aria-label="Delete model"
                          icon={<DeleteIcon />}
                          colorScheme="red"
                          variant="ghost"
                          onClick={() => handleDeleteModel(model.symbol, model.model_type)}
                          isLoading={deleteModelMutation.isLoading}
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

export default ModelTrainingWithAPI;
