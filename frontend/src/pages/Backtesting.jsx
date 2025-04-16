import React, { useState } from 'react';
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
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Badge,
  Input,
} from '@chakra-ui/react';

// Mock data
const mockStrategies = [
  {
    id: 'prediction_based',
    name: 'Prediction Based Strategy',
    description: 'Trading strategy based solely on model predictions'
  },
  {
    id: 'ma_prediction',
    name: 'Moving Average + Prediction Strategy',
    description: 'Trading strategy combining moving averages with model predictions'
  },
  {
    id: 'bollinger_bands',
    name: 'Bollinger Bands Strategy',
    description: 'Trading strategy using Bollinger Bands with model predictions'
  },
  {
    id: 'rsi_strategy',
    name: 'RSI Strategy',
    description: 'Trading strategy using RSI indicator with model predictions'
  }
];

const mockBacktestResults = [
  {
    symbol: 'BTC',
    model_type: 'lstm',
    start_date: '2024-10-01',
    end_date: '2025-04-01',
    initial_capital: 10000,
    final_value: 13500,
    roi: 35,
    sharpe_ratio: 1.8,
    max_drawdown: 12,
    win_rate: 65,
    trades: 42
  },
  {
    symbol: 'ETH',
    model_type: 'xgboost',
    start_date: '2024-10-01',
    end_date: '2025-04-01',
    initial_capital: 10000,
    final_value: 11200,
    roi: 12,
    sharpe_ratio: 1.2,
    max_drawdown: 15,
    win_rate: 58,
    trades: 37
  },
  {
    symbol: 'SOL',
    model_type: 'gru',
    start_date: '2024-10-01',
    end_date: '2025-04-01',
    initial_capital: 10000,
    final_value: 14200,
    roi: 42,
    sharpe_ratio: 1.9,
    max_drawdown: 18,
    win_rate: 62,
    trades: 45
  }
];

const Backtesting = () => {
  const [selectedCrypto, setSelectedCrypto] = useState('BTC');
  const [selectedModel, setSelectedModel] = useState('lstm');
  const [selectedStrategy, setSelectedStrategy] = useState('prediction_based');
  const [startDate, setStartDate] = useState('2024-10-01');
  const [endDate, setEndDate] = useState('2025-04-01');
  const [initialCapital, setInitialCapital] = useState(10000);
  const [positionSize, setPositionSize] = useState(0.1);
  const [isRunning, setIsRunning] = useState(false);
  const [currentResult, setCurrentResult] = useState(null);
  const toast = useToast();

  // Handle run backtest
  const handleRunBacktest = () => {
    setIsRunning(true);
    
    // Simulate API call
    setTimeout(() => {
      setIsRunning(false);
      
      // Create a mock result
      const result = {
        symbol: selectedCrypto,
        model_type: selectedModel,
        start_date: startDate,
        end_date: endDate,
        initial_capital: initialCapital,
        final_value: initialCapital * (1 + Math.random() * 0.5),
        roi: Math.random() * 50,
        sharpe_ratio: 1 + Math.random(),
        max_drawdown: 5 + Math.random() * 15,
        win_rate: 50 + Math.random() * 20,
        trades: 30 + Math.floor(Math.random() * 30)
      };
      
      setCurrentResult(result);
      
      toast({
        title: 'Backtest completed',
        description: `Backtest for ${selectedCrypto} using ${selectedModel.toUpperCase()} model completed successfully`,
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
    }, 2000);
  };

  return (
    <Box p={5}>
      <Heading mb={5}>Backtesting</Heading>
      
      <Tabs variant="enclosed">
        <TabList>
          <Tab>Run Backtest</Tab>
          <Tab>Backtest Results</Tab>
        </TabList>
        
        <TabPanels>
          {/* Run Backtest Tab */}
          <TabPanel>
            <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={5}>
              {/* Backtest Configuration Card */}
              <Card>
                <CardHeader>
                  <Heading size="md">Backtest Configuration</Heading>
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
                      <FormLabel>Model Type</FormLabel>
                      <Select 
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                      >
                        <option value="lstm">LSTM</option>
                        <option value="gru">GRU</option>
                        <option value="xgboost">XGBoost</option>
                        <option value="lightgbm">LightGBM</option>
                      </Select>
                    </FormControl>
                    
                    <FormControl>
                      <FormLabel>Trading Strategy</FormLabel>
                      <Select 
                        value={selectedStrategy}
                        onChange={(e) => setSelectedStrategy(e.target.value)}
                      >
                        {mockStrategies.map(strategy => (
                          <option key={strategy.id} value={strategy.id}>
                            {strategy.name}
                          </option>
                        ))}
                      </Select>
                    </FormControl>
                    
                    <FormControl>
                      <FormLabel>Start Date</FormLabel>
                      <Input 
                        type="date" 
                        value={startDate}
                        onChange={(e) => setStartDate(e.target.value)}
                      />
                    </FormControl>
                    
                    <FormControl>
                      <FormLabel>End Date</FormLabel>
                      <Input 
                        type="date" 
                        value={endDate}
                        onChange={(e) => setEndDate(e.target.value)}
                      />
                    </FormControl>
                    
                    <FormControl>
                      <FormLabel>Initial Capital</FormLabel>
                      <NumberInput 
                        value={initialCapital} 
                        onChange={(valueString) => setInitialCapital(parseInt(valueString))}
                        min={100}
                        max={1000000}
                      >
                        <NumberInputField />
                        <NumberInputStepper>
                          <NumberIncrementStepper />
                          <NumberDecrementStepper />
                        </NumberInputStepper>
                      </NumberInput>
                    </FormControl>
                    
                    <FormControl>
                      <FormLabel>Position Size (0-1)</FormLabel>
                      <NumberInput 
                        value={positionSize} 
                        onChange={(valueString) => setPositionSize(parseFloat(valueString))}
                        min={0.01}
                        max={1}
                        step={0.01}
                        precision={2}
                      >
                        <NumberInputField />
                        <NumberInputStepper>
                          <NumberIncrementStepper />
                          <NumberDecrementStepper />
                        </NumberInputStepper>
                      </NumberInput>
                    </FormControl>
                  </SimpleGrid>
                  
                  <Button 
                    mt={5} 
                    colorScheme="blue" 
                    onClick={handleRunBacktest}
                    isLoading={isRunning}
                    loadingText="Running..."
                    isDisabled={isRunning}
                  >
                    Run Backtest
                  </Button>
                </CardBody>
              </Card>
              
              {/* Backtest Results Card */}
              <Card>
                <CardHeader>
                  <Heading size="md">Current Backtest Result</Heading>
                </CardHeader>
                <CardBody>
                  {isRunning ? (
                    <Box textAlign="center" py={10}>
                      <Spinner size="xl" />
                      <Text mt={3}>Running backtest...</Text>
                    </Box>
                  ) : currentResult ? (
                    <SimpleGrid columns={{ base: 1, md: 2 }} spacing={5}>
                      <Stat>
                        <StatLabel>Final Value</StatLabel>
                        <StatNumber>${currentResult.final_value.toFixed(2)}</StatNumber>
                        <StatHelpText>
                          <StatArrow type={currentResult.roi > 0 ? 'increase' : 'decrease'} />
                          {currentResult.roi.toFixed(2)}%
                        </StatHelpText>
                      </Stat>
                      
                      <Stat>
                        <StatLabel>Sharpe Ratio</StatLabel>
                        <StatNumber>{currentResult.sharpe_ratio.toFixed(2)}</StatNumber>
                        <StatHelpText>
                          {currentResult.sharpe_ratio > 1 ? 'Good' : 'Poor'} risk-adjusted return
                        </StatHelpText>
                      </Stat>
                      
                      <Stat>
                        <StatLabel>Max Drawdown</StatLabel>
                        <StatNumber>{currentResult.max_drawdown.toFixed(2)}%</StatNumber>
                        <StatHelpText>
                          Maximum observed loss
                        </StatHelpText>
                      </Stat>
                      
                      <Stat>
                        <StatLabel>Win Rate</StatLabel>
                        <StatNumber>{currentResult.win_rate.toFixed(2)}%</StatNumber>
                        <StatHelpText>
                          {currentResult.trades} trades
                        </StatHelpText>
                      </Stat>
                      
                      <Box gridColumn="span 2">
                        <Text fontWeight="bold">Strategy:</Text>
                        <Text>
                          {mockStrategies.find(s => s.id === selectedStrategy)?.name || selectedStrategy}
                        </Text>
                        <Text mt={2} fontWeight="bold">Period:</Text>
                        <Text>
                          {currentResult.start_date} to {currentResult.end_date}
                        </Text>
                      </Box>
                    </SimpleGrid>
                  ) : (
                    <Alert status="info">
                      <AlertIcon />
                      No backtest results yet. Run a backtest to see results.
                    </Alert>
                  )}
                </CardBody>
              </Card>
            </SimpleGrid>
          </TabPanel>
          
          {/* Backtest Results Tab */}
          <TabPanel>
            <Card>
              <CardHeader>
                <Heading size="md">Historical Backtest Results</Heading>
              </CardHeader>
              <CardBody>
                {isRunning ? (
                  <Box textAlign="center" py={10}>
                    <Spinner size="xl" />
                    <Text mt={3}>Loading results...</Text>
                  </Box>
                ) : mockBacktestResults.length > 0 ? (
                  <Table variant="simple">
                    <Thead>
                      <Tr>
                        <Th>Symbol</Th>
                        <Th>Model</Th>
                        <Th>Period</Th>
                        <Th>Initial Capital</Th>
                        <Th>Final Value</Th>
                        <Th>ROI</Th>
                        <Th>Sharpe</Th>
                        <Th>Max DD</Th>
                        <Th>Win Rate</Th>
                      </Tr>
                    </Thead>
                    <Tbody>
                      {mockBacktestResults.map((result, index) => (
                        <Tr key={index}>
                          <Td>{result.symbol}</Td>
                          <Td>
                            <Badge colorScheme={
                              result.model_type === 'lstm' ? 'blue' :
                              result.model_type === 'gru' ? 'purple' :
                              result.model_type === 'xgboost' ? 'green' :
                              'orange'
                            }>
                              {result.model_type.toUpperCase()}
                            </Badge>
                          </Td>
                          <Td>{result.start_date} to {result.end_date}</Td>
                          <Td>${result.initial_capital}</Td>
                          <Td>${result.final_value}</Td>
                          <Td>
                            <Text color={result.roi > 0 ? 'green.500' : 'red.500'}>
                              {result.roi}%
                            </Text>
                          </Td>
                          <Td>{result.sharpe_ratio}</Td>
                          <Td>{result.max_drawdown}%</Td>
                          <Td>{result.win_rate}% ({result.trades} trades)</Td>
                        </Tr>
                      ))}
                    </Tbody>
                  </Table>
                ) : (
                  <Alert status="info">
                    <AlertIcon />
                    No historical backtest results available.
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

export default Backtesting;
