import React, { useState, useEffect } from 'react';
import {
  Box,
  Heading,
  Text,
  SimpleGrid,
  Select,
  Button,
  Flex,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  Card,
  CardHeader,
  CardBody,
  Stack,
  StackDivider,
  useToast,
  Spinner,
  Switch,
  FormControl,
  FormLabel,
} from '@chakra-ui/react';
import { useQuery } from 'react-query';
import axios from 'axios';

import PredictionChart from './PredictionChart';
import SimplePredictionChart from './SimplePredictionChart';
import PredictionHistoryChart from './PredictionHistoryChart';
import ExplanationView from './ExplanationView';

// Import configuration
import { API_BASE_URL } from '../config';

const Dashboard = () => {
  const [selectedCrypto, setSelectedCrypto] = useState('BTC');
  const [selectedModel, setSelectedModel] = useState('lstm');
  const [predictionHorizon, setPredictionHorizon] = useState(7);
  const [showConfidence, setShowConfidence] = useState(true);
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [useFeatureSelection, setUseFeatureSelection] = useState(true);
  const [useAnomalyDetection, setUseAnomalyDetection] = useState(true);
  const [useContinualLearning, setUseContinualLearning] = useState(false);
  const [useSimpleChart, setUseSimpleChart] = useState(true); // Use simple chart by default
  const toast = useToast();

  // Fetch prediction
  const { data: prediction, isLoading: predictionLoading, error: predictionError, refetch: refetchPrediction } = useQuery(
    ['prediction', selectedCrypto, selectedModel, predictionHorizon, showConfidence, useFeatureSelection, useAnomalyDetection, useContinualLearning],
    async () => {
      try {
        console.log('Making prediction request with:', {
          symbol: selectedCrypto,
          model_type: selectedModel,
          horizon: predictionHorizon,
          confidence_interval: showConfidence,
          use_feature_selection: useFeatureSelection,
          use_anomaly_detection: useAnomalyDetection,
          use_continual_learning: useContinualLearning
        });

        // Make the actual API call to the backend
        const axiosInstance = axios.create({
          baseURL: API_BASE_URL
        });

        const response = await axiosInstance.post('/api/predictions/predict', {
          symbol: selectedCrypto,
          model_type: selectedModel,
          horizon: predictionHorizon,
          confidence_interval: showConfidence,
          use_feature_selection: useFeatureSelection,
          use_anomaly_detection: useAnomalyDetection,
          use_continual_learning: useContinualLearning
        });

        console.log('Prediction response:', response.data);
        return response.data;
      } catch (error) {
        console.error('Error making prediction request:', error);
        throw error;
      }
    },
    {
      enabled: false, // Don't fetch on mount
      retry: 1,
      onError: (error) => {
        console.error('Query error handler:', error);
        toast({
          title: 'Error fetching prediction',
          description: error.response?.data?.detail || error.message,
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      }
    }
  );

  // Fetch explanation
  const { data: explanation, isLoading: explanationLoading, error: explanationError, refetch: refetchExplanation } = useQuery(
    ['explanation', selectedCrypto, selectedModel],
    async () => {
      try {
        console.log('Making explanation request with:', {
          symbol: selectedCrypto,
          model_type: selectedModel
        });

        // For development/testing, return mock data
        // Comment this out when backend is working
        return {
          symbol: selectedCrypto,
          model_type: selectedModel,
          prediction_date: new Date().toISOString(),
          feature_importance: [
            { feature: 'rsi_14', importance: 0.25 },
            { feature: 'macd_line', importance: 0.20 },
            { feature: 'bb_percent_b', importance: 0.15 },
            { feature: 'sma_50_200_cross', importance: 0.10 },
            { feature: 'volume', importance: 0.05 }
          ],
          shap_values: null
        };

        // Uncomment when backend is working
        /*
        const response = await axios.post('/api/predictions/explain', {
          symbol: selectedCrypto,
          model_type: selectedModel
        });
        return response.data;
        */
      } catch (error) {
        console.error('Error making explanation request:', error);
        throw error;
      }
    },
    {
      enabled: false, // Don't fetch on mount
      retry: 1,
      onError: (error) => {
        console.error('Explanation query error handler:', error);
        toast({
          title: 'Error fetching explanation',
          description: error.response?.data?.detail || error.message,
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      }
    }
  );

  const handlePredict = () => {
    refetchPrediction();
    refetchExplanation();
  };

  // Calculate price change if prediction is available
  const getPriceChange = () => {
    try {
      // Comprehensive validation
      if (!prediction) return null;
      if (!prediction.values) return null;
      if (!Array.isArray(prediction.values)) return null;
      if (prediction.values.length === 0) return null;

      const firstPrice = prediction.values[0];
      const lastPrice = prediction.values[prediction.values.length - 1];

      // Validate that prices are numbers
      if (typeof firstPrice !== 'number' || isNaN(firstPrice) ||
          typeof lastPrice !== 'number' || isNaN(lastPrice) ||
          firstPrice === 0) {
        console.error('Invalid price values:', { firstPrice, lastPrice });
        return null;
      }

      const change = ((lastPrice - firstPrice) / firstPrice) * 100;

      // Validate the change value
      if (isNaN(change) || !isFinite(change)) {
        console.error('Invalid change calculation:', change);
        return null;
      }

      return {
        value: change.toFixed(2),
        isPositive: change >= 0
      };
    } catch (error) {
      console.error('Error calculating price change:', error);
      return null;
    }
  };

  const priceChange = getPriceChange();

  return (
    <Box>
      <Heading mb={6}>Cryptocurrency Price Prediction Dashboard</Heading>

      {/* Controls */}
      <Card mb={6}>
        <CardBody>
          <Flex direction={{ base: 'column', md: 'row' }} gap={4} align="flex-end">
            <Box flex="1">
              <Text mb={2}>Cryptocurrency</Text>
              <Select
                value={selectedCrypto}
                onChange={(e) => setSelectedCrypto(e.target.value)}
              >
                <option value="BTC">Bitcoin (BTC)</option>
                <option value="ETH">Ethereum (ETH)</option>
                <option value="BNB">Binance Coin (BNB)</option>
                <option value="XRP">Ripple (XRP)</option>
                <option value="ADA">Cardano (ADA)</option>
              </Select>
            </Box>

            <Box flex="1">
              <Text mb={2}>Model Type</Text>
              <Select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
              >
                {/* Currently Supported Models */}
                <optgroup label="Basic Models">
                  <option value="lstm">LSTM</option>
                  <option value="gru">GRU</option>
                  <option value="xgboost">XGBoost</option>
                  <option value="lightgbm">LightGBM</option>
                </optgroup>

                {/* Bidirectional LSTM is supported */}
                <optgroup label="Advanced Models">
                  <option value="bidirectional_lstm">Bidirectional LSTM</option>
                </optgroup>

                {/* Other models will be enabled in future updates */}
              </Select>
            </Box>

            <Box flex="1">
              <Text mb={2}>Prediction Horizon (Days)</Text>
              <Select
                value={predictionHorizon}
                onChange={(e) => setPredictionHorizon(parseInt(e.target.value))}
              >
                <option value="3">3 Days</option>
                <option value="7">7 Days</option>
                <option value="14">14 Days</option>
                <option value="30">30 Days</option>
              </Select>
            </Box>

            <Box>
              <Stack spacing={2}>
                <Button
                  colorScheme="blue"
                  onClick={handlePredict}
                  isLoading={predictionLoading || explanationLoading}
                  loadingText="Predicting"
                >
                  Generate Prediction
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
                >
                  {showAdvancedOptions ? 'Hide Advanced Options' : 'Show Advanced Options'}
                </Button>
              </Stack>
            </Box>
          </Flex>

          {/* Advanced Options */}
          {showAdvancedOptions && (
            <Box mt={4} p={4} borderWidth="1px" borderRadius="md" bg="gray.50">
              <Text fontWeight="bold" mb={3}>Advanced Options</Text>
              <SimpleGrid columns={{ base: 1, md: 3 }} spacing={4}>
                <Box>
                  <Text mb={1} fontSize="sm">Feature Selection</Text>
                  <Select
                    size="sm"
                    value={useFeatureSelection ? "true" : "false"}
                    onChange={(e) => setUseFeatureSelection(e.target.value === "true")}
                  >
                    <option value="true">Enabled</option>
                    <option value="false">Disabled</option>
                  </Select>
                </Box>

                <Box>
                  <Text mb={1} fontSize="sm">Anomaly Detection</Text>
                  <Select
                    size="sm"
                    value={useAnomalyDetection ? "true" : "false"}
                    onChange={(e) => setUseAnomalyDetection(e.target.value === "true")}
                  >
                    <option value="true">Enabled</option>
                    <option value="false">Disabled</option>
                  </Select>
                </Box>

                <Box>
                  <Text mb={1} fontSize="sm">Continual Learning</Text>
                  <Select
                    size="sm"
                    value={useContinualLearning ? "true" : "false"}
                    onChange={(e) => setUseContinualLearning(e.target.value === "true")}
                  >
                    <option value="true">Enabled</option>
                    <option value="false">Disabled</option>
                  </Select>
                </Box>
              </SimpleGrid>

              <Text fontSize="xs" mt={3} color="gray.600">
                These advanced options can significantly improve prediction accuracy. Feature selection identifies the most important factors, anomaly detection handles outliers, and continual learning updates models with new data.
              </Text>
            </Box>
          )}
        </CardBody>
      </Card>

      {/* Prediction Results */}
      <SimpleGrid columns={{ base: 1, lg: 3 }} spacing={6} mb={6}>
        <Card gridColumn={{ lg: "span 2" }}>
          <CardHeader>
            <Heading size="md">Price Prediction</Heading>
          </CardHeader>
          <CardBody>
            {predictionLoading ? (
              <Flex justify="center" align="center" h="300px">
                <Spinner size="xl" />
              </Flex>
            ) : prediction ? (
              <>
                {useSimpleChart ? (
                  <SimplePredictionChart prediction={prediction} />
                ) : (
                  <PredictionChart prediction={prediction} />
                )}
                <Flex justify="flex-end" mt={2}>
                  <FormControl display="flex" alignItems="center" justifyContent="flex-end">
                    <FormLabel htmlFor="chart-toggle" mb="0" fontSize="sm">
                      Use simple chart
                    </FormLabel>
                    <Switch
                      id="chart-toggle"
                      isChecked={useSimpleChart}
                      onChange={() => setUseSimpleChart(!useSimpleChart)}
                    />
                  </FormControl>
                </Flex>
              </>
            ) : (
              <Flex justify="center" align="center" h="300px">
                <Text>Generate a prediction to see the chart</Text>
              </Flex>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader>
            <Heading size="md">Prediction Summary</Heading>
          </CardHeader>
          <CardBody>
            {predictionLoading ? (
              <Spinner />
            ) : prediction ? (
              <Stack divider={<StackDivider />} spacing={4}>
                <Stat>
                  <StatLabel>Current Price</StatLabel>
                  <StatNumber>
                    {(() => {
                      try {
                        if (!Array.isArray(prediction.values) || prediction.values.length === 0) {
                          return 'Not available';
                        }
                        const currentPrice = prediction.values[0];
                        if (typeof currentPrice !== 'number' || isNaN(currentPrice)) {
                          return 'Not available';
                        }
                        return `$${currentPrice.toLocaleString()}`;
                      } catch (error) {
                        console.error('Error formatting current price:', error);
                        return 'Not available';
                      }
                    })()}
                  </StatNumber>
                  <StatHelpText>
                    {(() => {
                      try {
                        if (!Array.isArray(prediction.timestamps) || prediction.timestamps.length === 0) {
                          return 'Date not available';
                        }
                        return new Date(prediction.timestamps[0]).toLocaleDateString();
                      } catch (error) {
                        console.error('Error formatting timestamp:', error);
                        return 'Date not available';
                      }
                    })()}
                  </StatHelpText>
                </Stat>

                <Stat>
                  <StatLabel>Predicted Price ({predictionHorizon} days)</StatLabel>
                  <StatNumber>
                    {(() => {
                      try {
                        if (!Array.isArray(prediction.values) || prediction.values.length === 0) {
                          return 'Not available';
                        }
                        const predictedPrice = prediction.values[prediction.values.length - 1];
                        if (typeof predictedPrice !== 'number' || isNaN(predictedPrice)) {
                          return 'Not available';
                        }
                        return `$${predictedPrice.toLocaleString()}`;
                      } catch (error) {
                        console.error('Error formatting predicted price:', error);
                        return 'Not available';
                      }
                    })()}
                  </StatNumber>
                  <StatHelpText>
                    {priceChange && (
                      <>
                        <StatArrow type={priceChange.isPositive ? 'increase' : 'decrease'} />
                        {priceChange.value}%
                      </>
                    )}
                  </StatHelpText>
                </Stat>

                <Stat>
                  <StatLabel>Confidence Range</StatLabel>
                  {prediction.confidence_lower && Array.isArray(prediction.confidence_lower) &&
                   prediction.confidence_lower.length > 0 &&
                   prediction.confidence_upper && Array.isArray(prediction.confidence_upper) &&
                   prediction.confidence_upper.length > 0 ? (
                    <>
                      <StatNumber>
                        {(() => {
                          try {
                            const lowerValue = prediction.confidence_lower[prediction.confidence_lower.length - 1];
                            const upperValue = prediction.confidence_upper[prediction.confidence_upper.length - 1];

                            // Validate values are numbers
                            if (typeof lowerValue !== 'number' || isNaN(lowerValue) ||
                                typeof upperValue !== 'number' || isNaN(upperValue)) {
                              return 'Calculation error';
                            }

                            return `$${lowerValue.toLocaleString()} - $${upperValue.toLocaleString()}`;
                          } catch (error) {
                            console.error('Error formatting confidence range:', error);
                            return 'Calculation error';
                          }
                        })()}
                      </StatNumber>
                      <StatHelpText>
                        Based on model uncertainty
                      </StatHelpText>
                    </>
                  ) : (
                    <StatHelpText>
                      Not available
                    </StatHelpText>
                  )}
                </Stat>
              </Stack>
            ) : (
              <Text>Generate a prediction to see the summary</Text>
            )}
          </CardBody>
        </Card>
      </SimpleGrid>

      {/* Prediction History & Learning */}
      <Card mb={5}>
        <CardHeader>
          <Heading size="md">Prediction History & Learning</Heading>
        </CardHeader>
        <CardBody>
          {selectedCrypto && selectedModel ? (
            <PredictionHistoryChart symbol={selectedCrypto} modelType={selectedModel} />
          ) : (
            <Text>Select a cryptocurrency and model to see prediction history</Text>
          )}
        </CardBody>
      </Card>

      {/* Explanation */}
      <Card>
        <CardHeader>
          <Heading size="md">Model Explanation</Heading>
        </CardHeader>
        <CardBody>
          {explanationLoading ? (
            <Spinner />
          ) : explanation ? (
            <ExplanationView explanation={explanation} />
          ) : (
            <Text>Generate a prediction to see the explanation</Text>
          )}
        </CardBody>
      </Card>
    </Box>
  );
};

export default Dashboard;
