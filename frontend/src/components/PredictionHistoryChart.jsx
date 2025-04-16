import React, { useState, useEffect } from 'react';
import {
  Box,
  Heading,
  Text,
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
  Badge,
  Select,
  Button,
  Flex,
  Spinner,
  Alert,
  AlertIcon,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  useToast,
} from '@chakra-ui/react';
import axios from 'axios';
import { API_BASE_URL } from '../config';
import Plot from 'react-plotly.js';

const PredictionHistoryChart = ({ symbol, modelType }) => {
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [metrics, setMetrics] = useState(null);
  const [selectedPrediction, setSelectedPrediction] = useState(null);
  const toast = useToast();

  // Fetch prediction history
  useEffect(() => {
    if (!symbol) return;

    const fetchPredictionHistory = async () => {
      setIsLoading(true);
      try {
        const response = await axios.get(
          `${API_BASE_URL}/api/feedback/history/${symbol}${modelType ? `?model_type=${modelType}` : ''}`
        );
        setPredictionHistory(response.data);
        
        // Select the most recent prediction by default
        if (response.data.length > 0) {
          setSelectedPrediction(response.data[0]);
        }
        
        setIsLoading(false);
      } catch (error) {
        console.error('Error fetching prediction history:', error);
        toast({
          title: 'Error fetching prediction history',
          description: error.message || 'An error occurred',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
        setIsLoading(false);
      }
    };

    const fetchMetrics = async () => {
      try {
        const response = await axios.get(
          `${API_BASE_URL}/api/feedback/metrics${symbol ? `?symbol=${symbol}` : ''}${modelType ? `&model_type=${modelType}` : ''}`
        );
        setMetrics(response.data);
      } catch (error) {
        console.error('Error fetching metrics:', error);
      }
    };

    fetchPredictionHistory();
    fetchMetrics();
  }, [symbol, modelType, toast]);

  // Process feedback
  const processFeedback = async () => {
    try {
      await axios.post(`${API_BASE_URL}/api/feedback/process`);
      toast({
        title: 'Feedback processing started',
        description: 'The system will process feedback for all predictions',
        status: 'info',
        duration: 3000,
        isClosable: true,
      });
    } catch (error) {
      console.error('Error processing feedback:', error);
      toast({
        title: 'Error processing feedback',
        description: error.message || 'An error occurred',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  // Format date
  const formatDate = (dateString) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  // Get performance badge color
  const getPerformanceBadgeColor = (r2) => {
    if (r2 >= 0.8) return 'green';
    if (r2 >= 0.6) return 'teal';
    if (r2 >= 0.4) return 'blue';
    if (r2 >= 0.2) return 'yellow';
    return 'red';
  };

  // Render prediction chart
  const renderPredictionChart = () => {
    if (!selectedPrediction) return null;

    const { prediction_values, prediction_timestamps, confidence_lower, confidence_upper, actual_values, actual_dates } = selectedPrediction;

    // Prepare data for chart
    const data = [
      {
        x: prediction_timestamps.map(ts => new Date(ts)),
        y: prediction_values,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Predicted',
        line: { color: 'blue' },
      },
    ];

    // Add confidence intervals if available
    if (confidence_lower && confidence_upper) {
      data.push({
        x: prediction_timestamps.map(ts => new Date(ts)),
        y: confidence_upper,
        type: 'scatter',
        mode: 'lines',
        name: 'Upper Bound',
        line: { color: 'rgba(0, 0, 255, 0.2)' },
        showlegend: false,
      });

      data.push({
        x: prediction_timestamps.map(ts => new Date(ts)),
        y: confidence_lower,
        type: 'scatter',
        mode: 'lines',
        name: 'Lower Bound',
        line: { color: 'rgba(0, 0, 255, 0.2)' },
        fill: 'tonexty',
        showlegend: false,
      });
    }

    // Add actual values if available
    if (actual_values && actual_dates) {
      data.push({
        x: actual_dates.map(ts => new Date(ts)),
        y: actual_values,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Actual',
        line: { color: 'green' },
      });
    }

    return (
      <Plot
        data={data}
        layout={{
          title: `${symbol} Price Prediction (${selectedPrediction.model_type.toUpperCase()})`,
          xaxis: { title: 'Date' },
          yaxis: { title: 'Price ($)' },
          autosize: true,
          height: 400,
          margin: { l: 50, r: 50, b: 50, t: 50, pad: 4 },
          legend: { orientation: 'h', y: -0.2 },
        }}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler={true}
        config={{ responsive: true }}
      />
    );
  };

  // Render prediction table
  const renderPredictionTable = () => {
    if (!selectedPrediction) return null;

    const { prediction_values, prediction_timestamps, confidence_lower, confidence_upper, actual_values, actual_dates } = selectedPrediction;

    // Prepare data for table
    const tableData = prediction_timestamps.map((timestamp, index) => {
      const row = {
        date: new Date(timestamp).toLocaleDateString(),
        predicted: prediction_values[index].toFixed(2),
      };

      if (confidence_lower && confidence_upper) {
        row.lower = confidence_lower[index].toFixed(2);
        row.upper = confidence_upper[index].toFixed(2);
      }

      // Find matching actual value if available
      if (actual_values && actual_dates) {
        const actualIndex = actual_dates.findIndex(d => {
          const predDate = new Date(timestamp);
          const actDate = new Date(d);
          return predDate.toDateString() === actDate.toDateString();
        });

        if (actualIndex !== -1) {
          row.actual = actual_values[actualIndex].toFixed(2);
          row.error = ((actual_values[actualIndex] - prediction_values[index]) / actual_values[actualIndex] * 100).toFixed(2);
        }
      }

      return row;
    });

    return (
      <Box overflowX="auto">
        <Table variant="simple" size="sm">
          <Thead>
            <Tr>
              <Th>Date</Th>
              <Th isNumeric>Predicted ($)</Th>
              {confidence_lower && confidence_upper && (
                <Th isNumeric>Confidence Range ($)</Th>
              )}
              {actual_values && actual_dates && (
                <>
                  <Th isNumeric>Actual ($)</Th>
                  <Th isNumeric>Error (%)</Th>
                </>
              )}
            </Tr>
          </Thead>
          <Tbody>
            {tableData.map((row, index) => (
              <Tr key={index}>
                <Td>{row.date}</Td>
                <Td isNumeric>${row.predicted}</Td>
                {confidence_lower && confidence_upper && (
                  <Td isNumeric>${row.lower} - ${row.upper}</Td>
                )}
                {actual_values && actual_dates && (
                  <>
                    <Td isNumeric>{row.actual ? `$${row.actual}` : '-'}</Td>
                    <Td isNumeric>
                      {row.error ? (
                        <Badge colorScheme={Math.abs(parseFloat(row.error)) < 5 ? 'green' : Math.abs(parseFloat(row.error)) < 10 ? 'yellow' : 'red'}>
                          {row.error}%
                        </Badge>
                      ) : '-'}
                    </Td>
                  </>
                )}
              </Tr>
            ))}
          </Tbody>
        </Table>
      </Box>
    );
  };

  // Render metrics
  const renderMetrics = () => {
    if (!metrics || !symbol || !modelType || !metrics[symbol] || !metrics[symbol][modelType]) {
      return (
        <Alert status="info">
          <AlertIcon />
          No performance metrics available yet. Metrics will be available after predictions are compared with actual values.
        </Alert>
      );
    }

    const modelMetrics = metrics[symbol][modelType];
    const lastRmse = modelMetrics.rmse.length > 0 ? modelMetrics.rmse[modelMetrics.rmse.length - 1] : null;
    const lastR2 = modelMetrics.r2.length > 0 ? modelMetrics.r2[modelMetrics.r2.length - 1] : null;
    const lastDirAcc = modelMetrics.directional_accuracy.length > 0 ? modelMetrics.directional_accuracy[modelMetrics.directional_accuracy.length - 1] : null;

    // Calculate trend
    const rmseChange = modelMetrics.rmse.length > 1 ? 
      modelMetrics.rmse[modelMetrics.rmse.length - 1] - modelMetrics.rmse[modelMetrics.rmse.length - 2] : 0;
    
    const r2Change = modelMetrics.r2.length > 1 ? 
      modelMetrics.r2[modelMetrics.r2.length - 1] - modelMetrics.r2[modelMetrics.r2.length - 2] : 0;
    
    const dirAccChange = modelMetrics.directional_accuracy.length > 1 ? 
      modelMetrics.directional_accuracy[modelMetrics.directional_accuracy.length - 1] - modelMetrics.directional_accuracy[modelMetrics.directional_accuracy.length - 2] : 0;

    return (
      <SimpleGrid columns={{ base: 1, md: 3 }} spacing={5}>
        <Stat>
          <StatLabel>RMSE (Root Mean Squared Error)</StatLabel>
          <StatNumber>{lastRmse !== null ? lastRmse.toFixed(4) : 'N/A'}</StatNumber>
          {rmseChange !== 0 && (
            <StatHelpText>
              <StatArrow type={rmseChange < 0 ? 'decrease' : 'increase'} />
              {Math.abs(rmseChange).toFixed(4)}
            </StatHelpText>
          )}
        </Stat>
        
        <Stat>
          <StatLabel>R² Score</StatLabel>
          <StatNumber>
            {lastR2 !== null ? (
              <Badge colorScheme={getPerformanceBadgeColor(lastR2)}>
                {(lastR2 * 100).toFixed(2)}%
              </Badge>
            ) : 'N/A'}
          </StatNumber>
          {r2Change !== 0 && (
            <StatHelpText>
              <StatArrow type={r2Change > 0 ? 'increase' : 'decrease'} />
              {Math.abs(r2Change * 100).toFixed(2)}%
            </StatHelpText>
          )}
        </Stat>
        
        <Stat>
          <StatLabel>Directional Accuracy</StatLabel>
          <StatNumber>
            {lastDirAcc !== null ? (
              <Badge colorScheme={lastDirAcc > 70 ? 'green' : lastDirAcc > 50 ? 'yellow' : 'red'}>
                {lastDirAcc.toFixed(2)}%
              </Badge>
            ) : 'N/A'}
          </StatNumber>
          {dirAccChange !== 0 && (
            <StatHelpText>
              <StatArrow type={dirAccChange > 0 ? 'increase' : 'decrease'} />
              {Math.abs(dirAccChange).toFixed(2)}%
            </StatHelpText>
          )}
        </Stat>
      </SimpleGrid>
    );
  };

  return (
    <Box>
      <Flex justifyContent="space-between" alignItems="center" mb={4}>
        <Heading size="md">Prediction History & Learning</Heading>
        <Button
          colorScheme="blue"
          size="sm"
          onClick={processFeedback}
        >
          Process Feedback & Learn
        </Button>
      </Flex>

      {isLoading ? (
        <Box textAlign="center" py={10}>
          <Spinner size="xl" />
          <Text mt={4}>Loading prediction history...</Text>
        </Box>
      ) : predictionHistory.length === 0 ? (
        <Alert status="info">
          <AlertIcon />
          No prediction history available for {symbol} {modelType ? `using ${modelType.toUpperCase()}` : ''}.
        </Alert>
      ) : (
        <>
          <Card mb={5}>
            <CardHeader>
              <Heading size="sm">Model Performance Metrics</Heading>
            </CardHeader>
            <CardBody>
              {renderMetrics()}
            </CardBody>
          </Card>

          <Card mb={5}>
            <CardHeader>
              <Flex justifyContent="space-between" alignItems="center">
                <Heading size="sm">Prediction Chart</Heading>
                {predictionHistory.length > 1 && (
                  <Select
                    value={selectedPrediction ? predictionHistory.indexOf(selectedPrediction) : 0}
                    onChange={(e) => setSelectedPrediction(predictionHistory[parseInt(e.target.value)])}
                    width="auto"
                    size="sm"
                  >
                    {predictionHistory.map((pred, index) => (
                      <option key={index} value={index}>
                        {formatDate(pred.created_at)}
                      </option>
                    ))}
                  </Select>
                )}
              </Flex>
            </CardHeader>
            <CardBody>
              {renderPredictionChart()}
            </CardBody>
          </Card>

          <Card>
            <CardHeader>
              <Heading size="sm">Prediction Data</Heading>
            </CardHeader>
            <CardBody>
              {renderPredictionTable()}
              
              {selectedPrediction && selectedPrediction.feedback_processed ? (
                <Alert status="success" mt={4}>
                  <AlertIcon />
                  This prediction has been processed for feedback and used to improve the model.
                </Alert>
              ) : (
                <Alert status="info" mt={4}>
                  <AlertIcon />
                  This prediction has not yet been processed for feedback. Click "Process Feedback & Learn" to update the model.
                </Alert>
              )}
            </CardBody>
          </Card>
        </>
      )}
    </Box>
  );
};

export default PredictionHistoryChart;
