import React, { useState } from 'react';
import {
  Box,
  Heading,
  Text,
  Button,
  Input,
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
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Spinner,
  Alert,
  AlertIcon,
  Progress,
  Flex,
  Badge,
  useColorModeValue,
} from '@chakra-ui/react';
import { keyframes } from '@emotion/react';
import { useQuery, useMutation } from 'react-query';
import axios from 'axios';

import { API_BASE_URL, AVAILABLE_CRYPTOCURRENCIES, AVAILABLE_DATA_SOURCES } from '../config';

// Define animations
const pulseAnimation = keyframes`
  0% { box-shadow: 0 0 0 0 rgba(66, 153, 225, 0.4); }
  70% { box-shadow: 0 0 0 10px rgba(66, 153, 225, 0); }
  100% { box-shadow: 0 0 0 0 rgba(66, 153, 225, 0); }
`;

const DataUpload = () => {
  const [selectedCrypto, setSelectedCrypto] = useState('BTC');
  const [selectedSource, setSelectedSource] = useState('coingecko');
  const [days, setDays] = useState(365);
  const [file, setFile] = useState(null);
  const [customSource, setCustomSource] = useState('');

  // Animation states
  const [fetchProgress, setFetchProgress] = useState(0);
  const [fetchStatus, setFetchStatus] = useState('idle'); // idle, fetching, success, error
  const [fetchMessage, setFetchMessage] = useState('');

  const toast = useToast();

  // Fetch data info
  const { data: dataInfo, isLoading: dataInfoLoading, error: dataInfoError, refetch: refetchDataInfo } = useQuery(
    ['dataInfo'],
    async () => {
      try {
        console.log('Fetching data info');
        // Add a baseURL to ensure the request goes to the right place
        const axiosInstance = axios.create({
          baseURL: API_BASE_URL
        });
        const response = await axiosInstance.get('/api/data/info');
        console.log('Data info response:', response.data);
        return response.data;
      } catch (error) {
        console.error('Error fetching data info:', error);
        throw error;
      }
    },
    {
      onError: (error) => {
        toast({
          title: 'Error fetching data info',
          description: error.response?.data?.detail || error.message,
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      }
    }
  );

  // Fetch data mutation
  const fetchDataMutation = useMutation(
    async (params) => {
      try {
        // Reset progress and set status to fetching
        setFetchProgress(0);
        setFetchStatus('fetching');
        setFetchMessage(`Fetching ${params.days} days of ${params.symbol} data from ${params.source}...`);

        // Simulate progress updates
        const progressInterval = setInterval(() => {
          setFetchProgress(prev => {
            const newProgress = prev + Math.random() * 15;
            return newProgress >= 90 ? 90 : newProgress; // Cap at 90% until complete
          });
        }, 500);

        // Make the actual API call
        console.log('Fetching data with params:', params);
        // Add a baseURL to ensure the request goes to the right place
        const axiosInstance = axios.create({
          baseURL: API_BASE_URL
        });
        const response = await axiosInstance.post('/api/data/fetch', params);

        // Clear the interval and set progress to 100%
        clearInterval(progressInterval);
        setFetchProgress(100);
        setFetchStatus('success');
        setFetchMessage(`Successfully fetched data for ${params.symbol}!`);

        return response.data;
      } catch (error) {
        console.error('Error fetching data:', error);
        // Set status to error
        setFetchStatus('error');
        setFetchMessage(`Error: ${error.response?.data?.detail || error.message}`);
        setFetchProgress(0);
        // Throw the error to be handled by onError
        throw error;
      }
    },
    {
      onSuccess: (data) => {
        console.log('Fetch data success:', data);
        toast({
          title: 'Data fetched successfully',
          description: `Fetched data for ${data.symbol} from ${data.source}`,
          status: 'success',
          duration: 5000,
          isClosable: true,
        });
        // Refresh the data info
        setTimeout(() => {
          refetchDataInfo();
          // Reset status after a delay
          setTimeout(() => {
            setFetchStatus('idle');
            setFetchMessage('');
          }, 3000);
        }, 1000);
      },
      onError: (error) => {
        console.error('Fetch data error:', error);
        toast({
          title: 'Error fetching data',
          description: error.response?.data?.detail || error.message,
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
        // Reset status after a delay
        setTimeout(() => {
          setFetchStatus('idle');
          setFetchMessage('');
        }, 3000);
      }
    }
  );

  // Upload data mutation
  const uploadDataMutation = useMutation(
    async ({ formData }) => {
      console.log('Uploading data');
      // Add a baseURL to ensure the request goes to the right place
      const axiosInstance = axios.create({
        baseURL: API_BASE_URL
      });
      const response = await axiosInstance.post('/api/data/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      console.log('Upload response:', response.data);
      return response.data;
    },
    {
      onSuccess: () => {
        toast({
          title: 'Data uploaded successfully',
          status: 'success',
          duration: 5000,
          isClosable: true,
        });
        refetchDataInfo();
        setFile(null);
      },
      onError: (error) => {
        toast({
          title: 'Error uploading data',
          description: error.response?.data?.detail || error.message,
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      }
    }
  );

  const handleFetchData = () => {
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

    if (!selectedSource) {
      toast({
        title: 'Please select a data source',
        status: 'warning',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    const parsedDays = parseInt(days);
    if (isNaN(parsedDays) || parsedDays <= 0 || parsedDays > 2000) {
      toast({
        title: 'Invalid number of days',
        description: 'Please enter a number between 1 and 2000',
        status: 'warning',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    // Call the mutation - the status updates are handled in the mutation
    fetchDataMutation.mutate({
      symbol: selectedCrypto,
      source: selectedSource,
      days: parsedDays
    });
  };

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUploadData = () => {
    if (!file) {
      toast({
        title: 'No file selected',
        status: 'warning',
        duration: 5000,
        isClosable: true,
      });
      return;
    }

    if (!customSource) {
      toast({
        title: 'Please enter a source name',
        status: 'warning',
        duration: 5000,
        isClosable: true,
      });
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('symbol', selectedCrypto);
    formData.append('source', customSource);

    uploadDataMutation.mutate({ formData });
  };

  return (
    <Box>
      <Heading mb={6}>Data Management</Heading>

      <Tabs variant="enclosed" mb={6}>
        <TabList>
          <Tab>Fetch Data</Tab>
          <Tab>Upload Data</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            <Card>
              <CardHeader>
                <Heading size="md">Fetch Cryptocurrency Data</Heading>
              </CardHeader>
              <CardBody>
                {/* Status animation */}
                {fetchStatus !== 'idle' && (
                  <Box mb={6} p={4} borderWidth="1px" borderRadius="lg"
                       bg={useColorModeValue(
                         fetchStatus === 'fetching' ? 'blue.50' :
                         fetchStatus === 'success' ? 'green.50' : 'red.50',
                         fetchStatus === 'fetching' ? 'blue.900' :
                         fetchStatus === 'success' ? 'green.900' : 'red.900'
                       )}
                       animation={fetchStatus === 'fetching' ? `${pulseAnimation} 2s infinite` : 'none'}
                       transition="all 0.3s ease"
                  >
                    <Flex align="center" mb={2}>
                      {fetchStatus === 'fetching' && (
                        <Spinner size="sm" color="blue.500" mr={2} />
                      )}
                      <Badge colorScheme={
                        fetchStatus === 'fetching' ? 'blue' :
                        fetchStatus === 'success' ? 'green' : 'red'
                      }>
                        {fetchStatus === 'fetching' ? 'Fetching Data' :
                         fetchStatus === 'success' ? 'Success' : 'Error'}
                      </Badge>
                    </Flex>
                    <Text fontSize="sm" mb={2}>{fetchMessage}</Text>
                    {fetchStatus === 'fetching' && (
                      <Progress
                        value={fetchProgress}
                        size="sm"
                        colorScheme="blue"
                        hasStripe
                        isAnimated
                        borderRadius="full"
                      />
                    )}
                  </Box>
                )}

                <SimpleGrid columns={{ base: 1, md: 3 }} spacing={4} mb={4}>
                  <FormControl>
                    <FormLabel>Cryptocurrency</FormLabel>
                    <Select
                      value={selectedCrypto}
                      onChange={(e) => setSelectedCrypto(e.target.value)}
                      isDisabled={fetchStatus === 'fetching'}
                    >
                      {AVAILABLE_CRYPTOCURRENCIES.map(crypto => (
                        <option key={crypto.value} value={crypto.value}>{crypto.label}</option>
                      ))}
                    </Select>
                  </FormControl>

                  <FormControl>
                    <FormLabel>Data Source</FormLabel>
                    <Select
                      value={selectedSource}
                      onChange={(e) => setSelectedSource(e.target.value)}
                      isDisabled={fetchStatus === 'fetching'}
                    >
                      {AVAILABLE_DATA_SOURCES.map(source => (
                        <option key={source.value} value={source.value}>{source.label}</option>
                      ))}
                    </Select>
                  </FormControl>

                  <FormControl>
                    <FormLabel>Days of History</FormLabel>
                    <Select
                      value={days}
                      onChange={(e) => setDays(e.target.value)}
                      isDisabled={fetchStatus === 'fetching'}
                    >
                      <option value="30">30 Days</option>
                      <option value="90">90 Days</option>
                      <option value="180">180 Days</option>
                      <option value="365">1 Year</option>
                      <option value="730">2 Years</option>
                    </Select>
                  </FormControl>
                </SimpleGrid>

                <Button
                  colorScheme={fetchStatus === 'fetching' ? 'gray' : 'blue'}
                  onClick={handleFetchData}
                  isLoading={fetchDataMutation.isLoading}
                  isDisabled={fetchStatus === 'fetching'}
                  loadingText="Fetching Data"
                >
                  Fetch Data
                </Button>
              </CardBody>
            </Card>
          </TabPanel>

          <TabPanel>
            <Card>
              <CardHeader>
                <Heading size="md">Upload Custom Data</Heading>
              </CardHeader>
              <CardBody>
                <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4} mb={4}>
                  <FormControl>
                    <FormLabel>Cryptocurrency</FormLabel>
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
                  </FormControl>

                  <FormControl>
                    <FormLabel>Source Name</FormLabel>
                    <Input
                      value={customSource}
                      onChange={(e) => setCustomSource(e.target.value)}
                      placeholder="e.g., custom_exchange"
                    />
                  </FormControl>
                </SimpleGrid>

                <FormControl mb={4}>
                  <FormLabel>CSV File</FormLabel>
                  <Input
                    type="file"
                    accept=".csv"
                    onChange={handleFileChange}
                  />
                  <Text fontSize="sm" mt={1} color="gray.500">
                    File must contain at least timestamp and price/close columns
                  </Text>
                </FormControl>

                <Button
                  colorScheme="blue"
                  onClick={handleUploadData}
                  isLoading={uploadDataMutation.isLoading}
                  isDisabled={!file}
                >
                  Upload Data
                </Button>
              </CardBody>
            </Card>
          </TabPanel>
        </TabPanels>
      </Tabs>

      <Card>
        <CardHeader>
          <Heading size="md">Available Data</Heading>
        </CardHeader>
        <CardBody>
          {dataInfoLoading ? (
            <Spinner />
          ) : dataInfoError ? (
            <Alert status="error">
              <AlertIcon />
              Error loading data information
            </Alert>
          ) : dataInfo && dataInfo.length > 0 ? (
            <Table variant="simple">
              <Thead>
                <Tr>
                  <Th>Symbol</Th>
                  <Th>Source</Th>
                  <Th>Date Range</Th>
                  <Th>Rows</Th>
                  <Th>Columns</Th>
                </Tr>
              </Thead>
              <Tbody>
                {dataInfo.map((info, index) => (
                  <Tr key={index}>
                    <Td>{info.symbol.toUpperCase()}</Td>
                    <Td>{info.source}</Td>
                    <Td>
                      {new Date(info.start_date).toLocaleDateString()} to{' '}
                      {new Date(info.end_date).toLocaleDateString()}
                    </Td>
                    <Td>{info.rows.toLocaleString()}</Td>
                    <Td>{info.columns.join(', ')}</Td>
                  </Tr>
                ))}
              </Tbody>
            </Table>
          ) : (
            <Text>No data available. Fetch or upload data to get started.</Text>
          )}
        </CardBody>
      </Card>
    </Box>
  );
};

export default DataUpload;
