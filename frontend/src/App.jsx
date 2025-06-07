import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { Box, ChakraProvider, useToast } from '@chakra-ui/react'

import Dashboard from './components/Dashboard'
import Navbar from './components/Navbar'

// Import components
import DataUpload from './components/DataUpload'
import DataManagement from './pages/DataManagement'
import MarketSentiment from './pages/MarketSentiment'
import AutoTraining from './pages/AutoTraining'
import ModelTrainingWithAPI from './components/ModelTrainingWithAPI'
import SimpleBacktesting from './components/SimpleBacktesting'
import ErrorBoundary from './components/ErrorBoundary'
import ApiKeys from './pages/ApiKeys'

// Component to handle route-level error boundaries
const RouteWithErrorBoundary = ({ component: Component, ...rest }) => {
  const handleReset = () => {
    // Force a refresh of the current page
    window.location.reload();
  };

  return (
    <ErrorBoundary onReset={handleReset}>
      <Component {...rest} />
    </ErrorBoundary>
  );
};

function App() {
  return (
    <ChakraProvider>
      <ErrorBoundary>
        <Router>
          <Box minH="100vh" bg="gray.50">
            <Navbar />
            <Box as="main" p={4}>
              <Routes>
                <Route path="/" element={<RouteWithErrorBoundary component={Dashboard} />} />
                <Route path="/data" element={<RouteWithErrorBoundary component={DataManagement} />} />
                <Route path="/sentiment" element={<RouteWithErrorBoundary component={MarketSentiment} />} />
                <Route path="/auto-training" element={<RouteWithErrorBoundary component={AutoTraining} />} />
                <Route path="/models" element={<RouteWithErrorBoundary component={ModelTrainingWithAPI} />} />
                <Route path="/backtest" element={<RouteWithErrorBoundary component={SimpleBacktesting} />} />
                <Route path="/api-keys" element={<RouteWithErrorBoundary component={ApiKeys} />} />
              </Routes>
            </Box>
          </Box>
        </Router>
      </ErrorBoundary>
    </ChakraProvider>
  )
}

export default App
