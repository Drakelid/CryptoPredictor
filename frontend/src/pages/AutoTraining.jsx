import React, { useState, useEffect } from 'react';
import {
  Box,
  Heading,
  Text,
  Button,
  SimpleGrid,
  Card,
  CardHeader,
  CardBody,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  Switch,
  FormControl,
  FormLabel,
  Select,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Alert,
  AlertIcon,
  Spinner,
  Progress,
  useToast,
  Flex,
} from '@chakra-ui/react';
import axios from 'axios';
import { API_BASE_URL } from '../config';

const AutoTraining = () => {
  const [selectedCrypto, setSelectedCrypto] = useState('BTC');
  const [selectedModel, setSelectedModel] = useState('lstm');
  const [trainingStatus, setTrainingStatus] = useState({});
  const [isLoading, setIsLoading] = useState(true);
  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [isForceTraining, setIsForceTraining] = useState(false);
  const [isForceEvaluating, setIsForceEvaluating] = useState(false);
  const [isForceUpdating, setIsForceUpdating] = useState(false);
  const [autoTrainingEnabled, setAutoTrainingEnabled] = useState(true);
  const toast = useToast();

  // Fetch training status
  const fetchTrainingStatus = async () => {
    setIsLoading(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/api/auto-training/status`);
      setTrainingStatus(response.data);
      setIsLoading(false);
    } catch (error) {
      console.error('Error fetching training status:', error);
      toast({
        title: 'Error fetching training status',
        description: error.message || 'An error occurred',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      setIsLoading(false);
    }
  };

  // Start auto-training
  const startAutoTraining = async () => {
    setIsStarting(true);
    try {
      await axios.post(`${API_BASE_URL}/api/auto-training/start`);
      toast({
        title: 'Auto-training started',
        description: 'The automatic training process has been started',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
      setAutoTrainingEnabled(true);
      fetchTrainingStatus();
    } catch (error) {
      console.error('Error starting auto-training:', error);
      toast({
        title: 'Error starting auto-training',
        description: error.message || 'An error occurred',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsStarting(false);
    }
  };

  // Stop auto-training
  const stopAutoTraining = async () => {
    setIsStopping(true);
    try {
      await axios.post(`${API_BASE_URL}/api/auto-training/stop`);
      toast({
        title: 'Auto-training stopped',
        description: 'The automatic training process has been stopped',
        status: 'info',
        duration: 3000,
        isClosable: true,
      });
      setAutoTrainingEnabled(false);
      fetchTrainingStatus();
    } catch (error) {
      console.error('Error stopping auto-training:', error);
      toast({
        title: 'Error stopping auto-training',
        description: error.message || 'An error occurred',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsStopping(false);
    }
  };

  // Force train model
  const forceTrainModel = async () => {
    setIsForceTraining(true);
    try {
      await axios.post(`${API_BASE_URL}/api/auto-training/force-train/${selectedCrypto}/${selectedModel}`);
      toast({
        title: 'Training started',
        description: `Training of ${selectedCrypto} model using ${selectedModel} has been started`,
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
      // Wait a bit before refreshing status to allow the backend to update
      setTimeout(fetchTrainingStatus, 2000);
    } catch (error) {
      console.error('Error forcing model training:', error);
      toast({
        title: 'Error forcing model training',
        description: error.message || 'An error occurred',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsForceTraining(false);
    }
  };

  // Force evaluate model
  const forceEvaluateModel = async () => {
    setIsForceEvaluating(true);
    try {
      await axios.post(`${API_BASE_URL}/api/auto-training/force-evaluate/${selectedCrypto}/${selectedModel}`);
      toast({
        title: 'Evaluation started',
        description: `Evaluation of ${selectedCrypto} model using ${selectedModel} has been started`,
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
      // Wait a bit before refreshing status to allow the backend to update
      setTimeout(fetchTrainingStatus, 2000);
    } catch (error) {
      console.error('Error forcing model evaluation:', error);
      toast({
        title: 'Error forcing model evaluation',
        description: error.message || 'An error occurred',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsForceEvaluating(false);
    }
  };

  // Force update model
  const forceUpdateModel = async () => {
    setIsForceUpdating(true);
    try {
      await axios.post(`${API_BASE_URL}/api/auto-training/force-update/${selectedCrypto}/${selectedModel}`);
      toast({
        title: 'Update started',
        description: `Update of ${selectedCrypto} model using ${selectedModel} has been started`,
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
      // Wait a bit before refreshing status to allow the backend to update
      setTimeout(fetchTrainingStatus, 2000);
    } catch (error) {
      console.error('Error forcing model update:', error);
      toast({
        title: 'Error forcing model update',
        description: error.message || 'An error occurred',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsForceUpdating(false);
    }
  };

  // Format date
  const formatDate = (dateString) => {
    if (!dateString) return 'Never';
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  // Get model status
  const getModelStatus = () => {
    if (!trainingStatus[selectedCrypto] || !trainingStatus[selectedCrypto][selectedModel]) {
      return null;
    }
    return trainingStatus[selectedCrypto][selectedModel];
  };

  // Get performance badge color
  const getPerformanceBadgeColor = (r2) => {
    if (r2 >= 0.8) return 'green';
    if (r2 >= 0.6) return 'teal';
    if (r2 >= 0.4) return 'blue';
    if (r2 >= 0.2) return 'yellow';
    return 'red';
  };

  // Fetch status on component mount and periodically
  useEffect(() => {
    fetchTrainingStatus();

    // Refresh status every 30 seconds
    const intervalId = setInterval(fetchTrainingStatus, 30000);

    return () => clearInterval(intervalId);
  }, []);

  // Get all cryptocurrencies from training status
  const getCryptocurrencies = () => {
    return Object.keys(trainingStatus).filter(key => key !== 'auto_training_enabled');
  };

  // Get all model types from training status
  const getModelTypes = () => {
    if (!trainingStatus[selectedCrypto]) return [];
    return Object.keys(trainingStatus[selectedCrypto]);
  };

  return (
    <Box p={5}>
      <Heading mb={5}>Automatic Model Training & Improvement</Heading>

      {isLoading ? (
        <Box textAlign="center" py={10}>
          <Spinner size="xl" />
          <Text mt={4}>Loading training status...</Text>
        </Box>
      ) : (
        <>
          <Card mb={5}>
            <CardHeader>
              <Heading size="md">Auto-Training Control</Heading>
            </CardHeader>
            <CardBody>
              <SimpleGrid columns={{ base: 1, md: 2 }} spacing={5} mb={5}>
                <Box>
                  <FormControl display="flex" alignItems="center">
                    <FormLabel htmlFor="auto-training-switch" mb="0">
                      Automatic Training
                    </FormLabel>
                    <Switch
                      id="auto-training-switch"
                      isChecked={autoTrainingEnabled}
                      onChange={() => autoTrainingEnabled ? stopAutoTraining() : startAutoTraining()}
                      colorScheme="blue"
                      size="lg"
                    />
                  </FormControl>
                  <Text fontSize="sm" color="gray.500" mt={2}>
                    When enabled, models will automatically train, evaluate, and improve over time.
                  </Text>
                </Box>

                <Box>
                  <Button
                    colorScheme="blue"
                    onClick={startAutoTraining}
                    isLoading={isStarting}
                    isDisabled={autoTrainingEnabled}
                    mr={3}
                  >
                    Start Auto-Training
                  </Button>
                  <Button
                    colorScheme="red"
                    onClick={stopAutoTraining}
                    isLoading={isStopping}
                    isDisabled={!autoTrainingEnabled}
                  >
                    Stop Auto-Training
                  </Button>
                </Box>
              </SimpleGrid>

              <Alert status="info">
                <AlertIcon />
                Auto-training runs in the background and automatically improves model accuracy over time.
                Models are trained with new data, evaluated regularly, and continuously updated.
              </Alert>
            </CardBody>
          </Card>

          <Card mb={5}>
            <CardHeader>
              <Heading size="md">Manual Control</Heading>
            </CardHeader>
            <CardBody>
              <SimpleGrid columns={{ base: 1, md: 2 }} spacing={5} mb={5}>
                <Box>
                  <Text mb={2}>Select Cryptocurrency</Text>
                  <Select
                    value={selectedCrypto}
                    onChange={(e) => setSelectedCrypto(e.target.value)}
                    mb={4}
                  >
                    {getCryptocurrencies().map((crypto) => (
                      <option key={crypto} value={crypto}>{crypto}</option>
                    ))}
                  </Select>
                </Box>

                <Box>
                  <Text mb={2}>Select Model Type</Text>
                  <Select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    mb={4}
                  >
                    {getModelTypes().map((model) => (
                      <option key={model} value={model}>{model.toUpperCase()}</option>
                    ))}
                  </Select>
                </Box>
              </SimpleGrid>

              <SimpleGrid columns={{ base: 1, md: 3 }} spacing={5}>
                <Button
                  colorScheme="blue"
                  onClick={forceTrainModel}
                  isLoading={isForceTraining}
                >
                  Train Model
                </Button>

                <Button
                  colorScheme="teal"
                  onClick={forceEvaluateModel}
                  isLoading={isForceEvaluating}
                >
                  Evaluate Model
                </Button>

                <Button
                  colorScheme="green"
                  onClick={forceUpdateModel}
                  isLoading={isForceUpdating}
                >
                  Update Model
                </Button>
              </SimpleGrid>
            </CardBody>
          </Card>

          <Tabs variant="enclosed">
            <TabList>
              <Tab>Model Status</Tab>
              <Tab>Performance Metrics</Tab>
              <Tab>Training History</Tab>
            </TabList>

            <TabPanels>
              {/* Model Status Tab */}
              <TabPanel>
                {getModelStatus() ? (
                  <Card>
                    <CardHeader>
                      <Heading size="md">{selectedCrypto} - {selectedModel.toUpperCase()} Model Status</Heading>
                    </CardHeader>
                    <CardBody>
                      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={5}>
                        <Stat>
                          <StatLabel>Model Exists</StatLabel>
                          <StatNumber>
                            <Badge colorScheme={getModelStatus().model_exists ? 'green' : 'red'}>
                              {getModelStatus().model_exists ? 'Yes' : 'No'}
                            </Badge>
                          </StatNumber>
                        </Stat>

                        <Stat>
                          <StatLabel>Last Training</StatLabel>
                          <StatNumber fontSize="md">
                            {formatDate(getModelStatus().last_training)}
                          </StatNumber>
                        </Stat>

                        <Stat>
                          <StatLabel>Last Evaluation</StatLabel>
                          <StatNumber fontSize="md">
                            {formatDate(getModelStatus().last_evaluation)}
                          </StatNumber>
                        </Stat>

                        <Stat>
                          <StatLabel>Last Update</StatLabel>
                          <StatNumber fontSize="md">
                            {formatDate(getModelStatus().last_update)}
                          </StatNumber>
                        </Stat>
                      </SimpleGrid>

                      {getModelStatus().performance && getModelStatus().performance.r2 !== undefined && (
                        <Box mt={5}>
                          <Text fontWeight="bold" mb={2}>Current Performance (R² Score)</Text>
                          <Progress
                            value={Math.max(0, Math.min(100, getModelStatus().performance.r2 * 100))}
                            colorScheme={getPerformanceBadgeColor(getModelStatus().performance.r2)}
                            size="lg"
                            mb={2}
                          />
                          <Text textAlign="center">
                            {(getModelStatus().performance.r2 * 100).toFixed(2)}% accuracy
                          </Text>
                        </Box>
                      )}
                    </CardBody>
                  </Card>
                ) : (
                  <Alert status="warning">
                    <AlertIcon />
                    No status information available for {selectedCrypto} using {selectedModel.toUpperCase()} model.
                  </Alert>
                )}
              </TabPanel>

              {/* Performance Metrics Tab */}
              <TabPanel>
                {getModelStatus() && getModelStatus().performance ? (
                  <Card>
                    <CardHeader>
                      <Heading size="md">{selectedCrypto} - {selectedModel.toUpperCase()} Performance Metrics</Heading>
                    </CardHeader>
                    <CardBody>
                      <SimpleGrid columns={{ base: 1, md: 3 }} spacing={5} mb={5}>
                        <Stat>
                          <StatLabel>RMSE (Root Mean Squared Error)</StatLabel>
                          <StatNumber>
                            {getModelStatus().performance.rmse ? getModelStatus().performance.rmse.toFixed(4) : 'N/A'}
                          </StatNumber>
                          <StatHelpText>Lower is better</StatHelpText>
                        </Stat>

                        <Stat>
                          <StatLabel>MAE (Mean Absolute Error)</StatLabel>
                          <StatNumber>
                            {getModelStatus().performance.mae ? getModelStatus().performance.mae.toFixed(4) : 'N/A'}
                          </StatNumber>
                          <StatHelpText>Lower is better</StatHelpText>
                        </Stat>

                        <Stat>
                          <StatLabel>R² Score</StatLabel>
                          <StatNumber>
                            {getModelStatus().performance.r2 ? getModelStatus().performance.r2.toFixed(4) : 'N/A'}
                          </StatNumber>
                          <StatHelpText>Higher is better (1.0 is perfect)</StatHelpText>
                        </Stat>
                      </SimpleGrid>

                      {getModelStatus().performance.accuracy_trend && getModelStatus().performance.accuracy_trend.length > 0 && (
                        <Box mt={5}>
                          <Text fontWeight="bold" mb={2}>Accuracy Trend (R² Score)</Text>
                          <Box height="100px" position="relative">
                            <Box
                              position="absolute"
                              bottom="0"
                              left="0"
                              right="0"
                              height="1px"
                              bg="gray.300"
                            />
                            <Flex height="100%" alignItems="flex-end">
                              {getModelStatus().performance.accuracy_trend.map((score, index) => (
                                <Box
                                  key={index}
                                  height={`${Math.max(0, Math.min(100, score * 100))}%`}
                                  width={`${100 / getModelStatus().performance.accuracy_trend.length}%`}
                                  bg={getPerformanceBadgeColor(score)}
                                  mx="1px"
                                  position="relative"
                                >
                                  <Text
                                    position="absolute"
                                    top="-20px"
                                    left="50%"
                                    transform="translateX(-50%)"
                                    fontSize="xs"
                                  >
                                    {(score * 100).toFixed(0)}%
                                  </Text>
                                </Box>
                              ))}
                            </Flex>
                          </Box>
                          <Text fontSize="sm" textAlign="center" mt={2}>
                            Accuracy trend over time (most recent on the right)
                          </Text>
                        </Box>
                      )}

                      <Alert status="info" mt={5}>
                        <AlertIcon />
                        These metrics show how well the model is performing. The auto-training system continuously works to improve these metrics over time.
                      </Alert>
                    </CardBody>
                  </Card>
                ) : (
                  <Alert status="warning">
                    <AlertIcon />
                    No performance metrics available for {selectedCrypto} using {selectedModel.toUpperCase()} model.
                  </Alert>
                )}
              </TabPanel>

              {/* Training History Tab */}
              <TabPanel>
                <Card>
                  <CardHeader>
                    <Heading size="md">Training History</Heading>
                  </CardHeader>
                  <CardBody>
                    <Table variant="simple">
                      <Thead>
                        <Tr>
                          <Th>Cryptocurrency</Th>
                          <Th>Model Type</Th>
                          <Th>Last Training</Th>
                          <Th>Last Evaluation</Th>
                          <Th>Last Update</Th>
                          <Th>Performance</Th>
                        </Tr>
                      </Thead>
                      <Tbody>
                        {getCryptocurrencies().map((crypto) => (
                          getModelTypes().map((model) => (
                            trainingStatus[crypto] && trainingStatus[crypto][model] ? (
                              <Tr key={`${crypto}-${model}`}>
                                <Td>{crypto}</Td>
                                <Td>{model.toUpperCase()}</Td>
                                <Td>{formatDate(trainingStatus[crypto][model].last_training)}</Td>
                                <Td>{formatDate(trainingStatus[crypto][model].last_evaluation)}</Td>
                                <Td>{formatDate(trainingStatus[crypto][model].last_update)}</Td>
                                <Td>
                                  {trainingStatus[crypto][model].performance && trainingStatus[crypto][model].performance.r2 !== undefined ? (
                                    <Badge colorScheme={getPerformanceBadgeColor(trainingStatus[crypto][model].performance.r2)}>
                                      {(trainingStatus[crypto][model].performance.r2 * 100).toFixed(2)}%
                                    </Badge>
                                  ) : (
                                    'N/A'
                                  )}
                                </Td>
                              </Tr>
                            ) : null
                          ))
                        ))}
                      </Tbody>
                    </Table>
                  </CardBody>
                </Card>
              </TabPanel>
            </TabPanels>
          </Tabs>
        </>
      )}
    </Box>
  );
};

export default AutoTraining;
