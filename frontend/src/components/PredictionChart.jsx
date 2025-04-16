import React, { useEffect, useRef, useState } from 'react';
import { Box, Text, Heading, Alert, AlertIcon, Code } from '@chakra-ui/react';

// Import Plotly directly if CDN fails
import Plotly from 'plotly.js-dist';

const PredictionChart = ({ prediction }) => {
  const [chartError, setChartError] = useState(null);
  const chartRef = useRef(null);

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

  // Log prediction data for debugging
  console.log('Prediction data:', JSON.stringify(prediction, null, 2));

  useEffect(() => {
    // Only run this effect if we have a valid chartRef and prediction data
    if (!chartRef.current || !prediction.values || !prediction.timestamps) {
      console.warn('Chart ref or prediction data not available');
      return;
    }

    try {
      // Format dates for display
      const formattedDates = prediction.timestamps.map(timestamp => {
        return new Date(timestamp).toLocaleDateString();
      });

      // Create the main prediction trace
      const predictionTrace = {
        x: formattedDates,
        y: prediction.values,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Prediction',
        line: { color: '#3182CE', width: 3 },
        marker: { size: 8 }
      };

      // Create traces array
      const traces = [predictionTrace];

      // Add confidence intervals if available
      if (prediction.confidence_lower && prediction.confidence_upper) {
        traces.push({
          x: [...formattedDates, ...formattedDates.slice().reverse()],
          y: [...prediction.confidence_upper, ...prediction.confidence_lower.slice().reverse()],
          fill: 'toself',
          fillcolor: 'rgba(49, 130, 206, 0.2)',
          line: { color: 'transparent' },
          name: 'Confidence Interval',
          showlegend: false,
          hoverinfo: 'skip'
        });
      }

      // Chart layout
      const layout = {
        title: `${prediction.symbol || 'Crypto'} Price Prediction (${(prediction.model_type || 'Model').toUpperCase()})`,
        xaxis: { title: 'Date' },
        yaxis: { title: 'Price ($)' },
        margin: { l: 50, r: 50, b: 50, t: 80, pad: 4 },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white'
      };

      // Chart config
      const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
      };

      // Use the imported Plotly directly
      console.log('Rendering chart with Plotly...');
      Plotly.newPlot(chartRef.current, traces, layout, config);
      console.log('Chart rendered successfully');
      setChartError(null);

    } catch (error) {
      console.error('Error rendering chart:', error);
      setChartError(error.message);
    }

    // Cleanup function
    return () => {
      if (chartRef.current) {
        try {
          Plotly.purge(chartRef.current);
        } catch (error) {
          console.error('Error cleaning up chart:', error);
        }
      }
    };
  }, [prediction]); // Re-render when prediction changes

  return (
    <Box p={4} bg="white" borderRadius="lg" boxShadow="sm" w="100%" h="400px">
      {chartError ? (
        <Alert status="error">
          <AlertIcon />
          Error rendering chart: {chartError}
        </Alert>
      ) : (
        <div ref={chartRef} style={{ width: '100%', height: '100%' }} />
      )}
    </Box>
  );
};

export default PredictionChart;
