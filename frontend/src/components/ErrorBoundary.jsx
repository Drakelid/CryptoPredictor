import React from 'react';
import { Box, Heading, Text, Button, VStack, Code, useToast } from '@chakra-ui/react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { 
      hasError: false,
      error: null,
      errorInfo: null
    };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    // Log the error to console
    console.error('Error caught by ErrorBoundary:', error, errorInfo);
    this.setState({ errorInfo });
    
    // You can also log the error to an error reporting service
    // logErrorToService(error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
    // Attempt to recover by resetting the component state
    if (this.props.onReset) {
      this.props.onReset();
    }
  }

  render() {
    if (this.state.hasError) {
      // Render fallback UI
      return (
        <Box 
          p={5} 
          m={5} 
          borderWidth="1px" 
          borderRadius="lg" 
          bg="red.50" 
          borderColor="red.200"
        >
          <VStack spacing={4} align="stretch">
            <Heading size="md" color="red.500">Something went wrong</Heading>
            <Text>The application encountered an unexpected error. Please try again.</Text>
            
            {this.state.error && (
              <Box bg="gray.50" p={3} borderRadius="md">
                <Text fontWeight="bold">Error:</Text>
                <Code colorScheme="red" whiteSpace="pre-wrap">
                  {this.state.error.toString()}
                </Code>
              </Box>
            )}
            
            <Button 
              colorScheme="blue" 
              onClick={this.handleReset}
              size="sm"
              alignSelf="flex-start"
            >
              Try Again
            </Button>
          </VStack>
        </Box>
      );
    }

    // If no error, render children normally
    return this.props.children;
  }
}

export default ErrorBoundary;
