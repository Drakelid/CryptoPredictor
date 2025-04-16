import React from 'react';
import { Box, Text, Alert, AlertIcon } from '@chakra-ui/react';

/**
 * A simple chart component that displays prediction data as a table
 * This is a fallback for when Plotly doesn't work
 */
const SimplePredictionChart = ({ prediction }) => {
  // Validate prediction data
  if (!prediction) {
    return <Box><Alert status="info"><AlertIcon />No prediction data available</Alert></Box>;
  }

  if (!prediction.values || !Array.isArray(prediction.values) || prediction.values.length === 0) {
    return <Box><Alert status="info"><AlertIcon />Prediction values are missing or invalid</Alert></Box>;
  }

  if (!prediction.timestamps || !Array.isArray(prediction.timestamps) || prediction.timestamps.length === 0) {
    return <Box><Alert status="info"><AlertIcon />Prediction timestamps are missing or invalid</Alert></Box>;
  }

  // Format dates for display
  const formattedData = prediction.timestamps.map((timestamp, index) => {
    return {
      date: new Date(timestamp).toLocaleDateString(),
      price: prediction.values[index].toFixed(2),
      lower: prediction.confidence_lower ? prediction.confidence_lower[index].toFixed(2) : null,
      upper: prediction.confidence_upper ? prediction.confidence_upper[index].toFixed(2) : null
    };
  });

  return (
    <Box p={4} bg="white" borderRadius="lg" boxShadow="sm" w="100%" overflowX="auto">
      <Text fontSize="xl" fontWeight="bold" mb={4}>
        {prediction.symbol} Price Prediction ({prediction.model_type.toUpperCase()})
      </Text>
      
      <Box as="table" width="100%" style={{ borderCollapse: 'collapse' }}>
        <Box as="thead" bg="blue.50">
          <Box as="tr">
            <Box as="th" p={2} textAlign="left" borderBottom="1px solid" borderColor="gray.200">Date</Box>
            <Box as="th" p={2} textAlign="right" borderBottom="1px solid" borderColor="gray.200">Price ($)</Box>
            {prediction.confidence_lower && prediction.confidence_upper && (
              <Box as="th" p={2} textAlign="right" borderBottom="1px solid" borderColor="gray.200">Confidence Range</Box>
            )}
          </Box>
        </Box>
        <Box as="tbody">
          {formattedData.map((item, index) => (
            <Box 
              as="tr" 
              key={index}
              bg={index === formattedData.length - 1 ? "blue.50" : "white"}
              fontWeight={index === formattedData.length - 1 ? "bold" : "normal"}
            >
              <Box as="td" p={2} borderBottom="1px solid" borderColor="gray.200">{item.date}</Box>
              <Box as="td" p={2} textAlign="right" borderBottom="1px solid" borderColor="gray.200">${item.price}</Box>
              {prediction.confidence_lower && prediction.confidence_upper && (
                <Box as="td" p={2} textAlign="right" borderBottom="1px solid" borderColor="gray.200">
                  ${item.lower} - ${item.upper}
                </Box>
              )}
            </Box>
          ))}
        </Box>
      </Box>
    </Box>
  );
};

export default SimplePredictionChart;
