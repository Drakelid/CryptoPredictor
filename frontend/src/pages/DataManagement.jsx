import React, { useState, useEffect, useRef } from 'react';
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
  Switch,
} from '@chakra-ui/react';

// Mock data
const mockDataInfo = [
  {
    symbol: 'BTC',
    source: 'coingecko',
    start_date: '2024-04-04',
    end_date: '2025-04-04',
    rows: 365,
    columns: ['timestamp', 'price', 'volume', 'market_cap']
  },
  {
    symbol: 'ETH',
    source: 'coingecko',
    start_date: '2024-04-04',
    end_date: '2025-04-04',
    rows: 365,
    columns: ['timestamp', 'price', 'volume', 'market_cap']
  },
  {
    symbol: 'SOL',
    source: 'coingecko',
    start_date: '2024-04-04',
    end_date: '2025-04-04',
    rows: 365,
    columns: ['timestamp', 'price', 'volume', 'market_cap']
  }
];

const DataManagement = () => {
  const [selectedCrypto, setSelectedCrypto] = useState('BTC');
  const [selectedSource, setSelectedSource] = useState('coingecko');
  const [days, setDays] = useState(365);
  const [file, setFile] = useState(null);
  const [customSource, setCustomSource] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [lastFetchTime, setLastFetchTime] = useState(null);
  const [autoFetchEnabled, setAutoFetchEnabled] = useState(true);
  const [nextFetchTime, setNextFetchTime] = useState(null);
  const intervalRef = useRef(null);
  const toast = useToast();

  // List of cryptocurrencies to fetch
  const cryptocurrencies = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL'];

  // Function to fetch data for a single cryptocurrency
  const fetchDataForCrypto = (crypto) => {
    console.log(`Fetching data for ${crypto} from ${selectedSource}...`);

    // Simulate API call for a single cryptocurrency
    return new Promise((resolve) => {
      setTimeout(() => {
        console.log(`Fetched data for ${crypto}`);
        resolve({
          symbol: crypto,
          source: selectedSource,
          days: days
        });
      }, 500); // Short timeout for simulation
    });
  };

  // Function to fetch data for all cryptocurrencies
  const fetchAllCryptoData = async () => {
    setIsLoading(true);
    setLastFetchTime(new Date());
    setNextFetchTime(new Date(Date.now() + 2 * 60 * 60 * 1000)); // 2 hours from now

    try {
      // Process cryptocurrencies sequentially to avoid rate limiting
      for (const crypto of cryptocurrencies) {
        await fetchDataForCrypto(crypto);
      }

      toast({
        title: 'Data fetched successfully',
        description: `Fetched data for all cryptocurrencies from ${selectedSource}`,
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
    } catch (error) {
      console.error('Error fetching data:', error);
      toast({
        title: 'Error fetching data',
        description: error.message || 'An error occurred while fetching data',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Handle manual fetch data button click
  const handleFetchData = () => {
    setIsLoading(true);

    // Simulate API call for a single cryptocurrency
    setTimeout(() => {
      setIsLoading(false);
      setLastFetchTime(new Date());
      toast({
        title: 'Data fetched successfully',
        description: `Fetched ${days} days of data for ${selectedCrypto} from ${selectedSource}`,
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
    }, 1500);
  };

  // Handle file upload
  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  // Set up automatic data fetching
  useEffect(() => {
    // Fetch data immediately when the component mounts
    if (autoFetchEnabled) {
      fetchAllCryptoData();
    }

    // Set up interval for automatic fetching (every 2 hours)
    const twoHoursInMs = 2 * 60 * 60 * 1000;
    intervalRef.current = setInterval(() => {
      if (autoFetchEnabled) {
        console.log('Auto-fetching data for all cryptocurrencies...');
        fetchAllCryptoData();
      }
    }, twoHoursInMs);

    // Update the countdown timer every minute
    const countdownInterval = setInterval(() => {
      if (nextFetchTime) {
        const now = new Date();
        const timeRemaining = nextFetchTime.getTime() - now.getTime();
        if (timeRemaining <= 0) {
          setNextFetchTime(new Date(Date.now() + twoHoursInMs));
        }
      }
    }, 60000); // Update every minute

    // Clean up intervals on component unmount
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      clearInterval(countdownInterval);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoFetchEnabled]); // Re-run effect when autoFetchEnabled changes

  // Toggle automatic fetching
  const toggleAutoFetch = () => {
    setAutoFetchEnabled(!autoFetchEnabled);
  };

  // Handle upload data
  const handleUploadData = () => {
    if (!file) {
      toast({
        title: 'No file selected',
        description: 'Please select a file to upload',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      return;
    }

    setIsLoading(true);

    // Simulate API call
    setTimeout(() => {
      setIsLoading(false);
      toast({
        title: 'Data uploaded successfully',
        description: `Uploaded data for ${selectedCrypto} from ${file.name}`,
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
      setFile(null);
    }, 1500);
  };

  return (
    <Box p={5}>
      <Heading mb={5}>Data Management</Heading>

      <Tabs variant="enclosed">
        <TabList>
          <Tab>Fetch Data</Tab>
          <Tab>Upload Data</Tab>
          <Tab>Available Data</Tab>
        </TabList>

        <TabPanels>
          {/* Fetch Data Tab */}
          <TabPanel>
            <Card>
              <CardHeader>
                <Heading size="md">Fetch Cryptocurrency Data</Heading>
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
                    <FormLabel>Data Source</FormLabel>
                    <Select
                      value={selectedSource}
                      onChange={(e) => setSelectedSource(e.target.value)}
                    >
                      <option value="coingecko">CoinGecko</option>
                      <option value="binance">Binance</option>
                      <option value="coinmarketcap">CoinMarketCap</option>
                    </Select>
                  </FormControl>

                  <FormControl>
                    <FormLabel>Days of Historical Data</FormLabel>
                    <Input
                      type="number"
                      value={days}
                      onChange={(e) => setDays(e.target.value)}
                      min={1}
                      max={2000}
                    />
                  </FormControl>
                </SimpleGrid>

                <Box mt={5}>
                  <Button
                    colorScheme="blue"
                    onClick={handleFetchData}
                    isLoading={isLoading}
                    loadingText="Fetching..."
                    mr={4}
                  >
                    Fetch Selected Crypto
                  </Button>

                  <Button
                    colorScheme="green"
                    onClick={fetchAllCryptoData}
                    isLoading={isLoading}
                    loadingText="Fetching All..."
                  >
                    Fetch All Cryptocurrencies
                  </Button>
                </Box>

                {/* Auto-fetch status and controls */}
                <Box mt={4} p={4} borderWidth="1px" borderRadius="md" bg="gray.50">
                  <Heading size="sm" mb={2}>Automatic Data Fetching</Heading>

                  <FormControl display="flex" alignItems="center" mb={3}>
                    <FormLabel htmlFor="auto-fetch-toggle" mb="0">
                      Enable automatic fetching every 2 hours
                    </FormLabel>
                    <Switch
                      id="auto-fetch-toggle"
                      isChecked={autoFetchEnabled}
                      onChange={toggleAutoFetch}
                      colorScheme="blue"
                    />
                  </FormControl>

                  {lastFetchTime && (
                    <Text fontSize="sm">
                      Last fetch: {lastFetchTime.toLocaleString()}
                    </Text>
                  )}

                  {nextFetchTime && autoFetchEnabled && (
                    <Text fontSize="sm">
                      Next automatic fetch: {nextFetchTime.toLocaleString()}
                    </Text>
                  )}

                  {!autoFetchEnabled && (
                    <Alert status="info" size="sm" mt={2}>
                      <AlertIcon />
                      Automatic fetching is disabled. Enable it to keep your data up to date.
                    </Alert>
                  )}
                </Box>
              </CardBody>
            </Card>
          </TabPanel>

          {/* Upload Data Tab */}
          <TabPanel>
            <Card>
              <CardHeader>
                <Heading size="md">Upload Cryptocurrency Data</Heading>
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
                    <FormLabel>Data Source</FormLabel>
                    <Input
                      value={customSource}
                      onChange={(e) => setCustomSource(e.target.value)}
                      placeholder="Enter custom source name"
                    />
                  </FormControl>

                  <FormControl>
                    <FormLabel>CSV File</FormLabel>
                    <Input
                      type="file"
                      accept=".csv"
                      onChange={handleFileChange}
                    />
                    <Text fontSize="sm" color="gray.500" mt={1}>
                      CSV file should contain columns: timestamp, price, volume, etc.
                    </Text>
                  </FormControl>
                </SimpleGrid>

                <Button
                  mt={5}
                  colorScheme="blue"
                  onClick={handleUploadData}
                  isLoading={isLoading}
                  loadingText="Uploading..."
                >
                  Upload Data
                </Button>
              </CardBody>
            </Card>
          </TabPanel>

          {/* Available Data Tab */}
          <TabPanel>
            <Card>
              <CardHeader>
                <Heading size="md">Available Data</Heading>
              </CardHeader>
              <CardBody>
                {isLoading ? (
                  <Box textAlign="center" py={10}>
                    <Spinner size="xl" />
                    <Text mt={3}>Loading data...</Text>
                  </Box>
                ) : mockDataInfo.length > 0 ? (
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
                      {mockDataInfo.map((item, index) => (
                        <Tr key={index}>
                          <Td>{item.symbol}</Td>
                          <Td>{item.source}</Td>
                          <Td>{item.start_date} to {item.end_date}</Td>
                          <Td>{item.rows}</Td>
                          <Td>{item.columns.join(', ')}</Td>
                        </Tr>
                      ))}
                    </Tbody>
                  </Table>
                ) : (
                  <Alert status="info">
                    <AlertIcon />
                    No data available. Please fetch or upload data first.
                  </Alert>
                )}
              </CardBody>
            </Card>
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Box>
  );
};

export default DataManagement;
