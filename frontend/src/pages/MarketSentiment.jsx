import React, { useState, useEffect } from 'react';
import {
  Box,
  Heading,
  Text,
  SimpleGrid,
  Card,
  CardHeader,
  CardBody,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  Progress,
  Select,
  Button,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  Spinner,
  Alert,
  AlertIcon,
  useToast,
} from '@chakra-ui/react';
import axios from 'axios';
import { API_BASE_URL } from '../config';

// Mock sentiment data
const mockSentimentData = {
  BTC: {
    social_sentiment: {
      twitter: 0.65,
      reddit: 0.58,
      overall: 0.62,
      change_24h: 0.05,
    },
    news_sentiment: {
      positive_articles: 28,
      negative_articles: 12,
      neutral_articles: 15,
      overall: 0.72,
      change_24h: -0.03,
    },
    fear_greed_index: {
      value: 65,
      classification: 'Greed',
      change_24h: 5,
    },
    market_indicators: {
      volume_change: 0.12,
      volatility: 0.08,
      momentum: 0.65,
    },
    historical: [
      { date: '2025-04-01', social: 0.57, news: 0.75, fear_greed: 60 },
      { date: '2025-04-02', social: 0.59, news: 0.73, fear_greed: 62 },
      { date: '2025-04-03', social: 0.61, news: 0.70, fear_greed: 63 },
      { date: '2025-04-04', social: 0.62, news: 0.72, fear_greed: 65 },
    ]
  },
  ETH: {
    social_sentiment: {
      twitter: 0.72,
      reddit: 0.68,
      overall: 0.70,
      change_24h: 0.02,
    },
    news_sentiment: {
      positive_articles: 32,
      negative_articles: 8,
      neutral_articles: 12,
      overall: 0.78,
      change_24h: 0.04,
    },
    fear_greed_index: {
      value: 72,
      classification: 'Greed',
      change_24h: 3,
    },
    market_indicators: {
      volume_change: 0.08,
      volatility: 0.06,
      momentum: 0.72,
    },
    historical: [
      { date: '2025-04-01', social: 0.68, news: 0.74, fear_greed: 69 },
      { date: '2025-04-02', social: 0.69, news: 0.76, fear_greed: 70 },
      { date: '2025-04-03', social: 0.70, news: 0.77, fear_greed: 71 },
      { date: '2025-04-04', social: 0.70, news: 0.78, fear_greed: 72 },
    ]
  }
};

const MarketSentiment = () => {
  const [selectedCrypto, setSelectedCrypto] = useState('BTC');
  const [sentimentData, setSentimentData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const toast = useToast();

  // Fetch sentiment data
  const fetchSentimentData = async () => {
    setIsLoading(true);
    try {
      // In a real app, this would be an API call
      // const response = await axios.get(`${API_BASE_URL}/api/sentiment/${selectedCrypto}`);
      // setSentimentData(response.data);
      
      // Using mock data for now
      setTimeout(() => {
        setSentimentData(mockSentimentData[selectedCrypto]);
        setLastUpdated(new Date());
        setIsLoading(false);
      }, 1000);
    } catch (error) {
      console.error('Error fetching sentiment data:', error);
      toast({
        title: 'Error fetching sentiment data',
        description: error.message || 'An error occurred',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      setIsLoading(false);
    }
  };

  // Refresh sentiment data
  const refreshSentimentData = async () => {
    setIsRefreshing(true);
    try {
      // In a real app, this would be an API call
      // const response = await axios.get(`${API_BASE_URL}/api/sentiment/${selectedCrypto}/refresh`);
      // setSentimentData(response.data);
      
      // Using mock data for now
      setTimeout(() => {
        // Simulate a small change in the data
        const updatedData = { ...mockSentimentData[selectedCrypto] };
        updatedData.social_sentiment.overall += (Math.random() * 0.1 - 0.05);
        updatedData.news_sentiment.overall += (Math.random() * 0.1 - 0.05);
        updatedData.fear_greed_index.value += Math.floor(Math.random() * 6 - 3);
        
        setSentimentData(updatedData);
        setLastUpdated(new Date());
        setIsRefreshing(false);
        
        toast({
          title: 'Sentiment data refreshed',
          description: `Latest sentiment data for ${selectedCrypto} has been loaded`,
          status: 'success',
          duration: 3000,
          isClosable: true,
        });
      }, 1500);
    } catch (error) {
      console.error('Error refreshing sentiment data:', error);
      toast({
        title: 'Error refreshing sentiment data',
        description: error.message || 'An error occurred',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      setIsRefreshing(false);
    }
  };

  // Fetch data when component mounts or selectedCrypto changes
  useEffect(() => {
    fetchSentimentData();
  }, [selectedCrypto]);

  // Helper function to render sentiment score with color
  const renderSentimentScore = (score) => {
    let color = 'gray.500';
    if (score > 0.7) color = 'green.500';
    else if (score > 0.5) color = 'green.300';
    else if (score > 0.4) color = 'yellow.400';
    else if (score > 0.3) color = 'orange.400';
    else color = 'red.500';

    return (
      <Text color={color} fontWeight="bold">
        {(score * 100).toFixed(1)}%
      </Text>
    );
  };

  // Helper function to render fear & greed index
  const renderFearGreedIndex = (value) => {
    let color = 'gray.500';
    let label = 'Neutral';
    
    if (value >= 75) {
      color = 'red.500';
      label = 'Extreme Greed';
    } else if (value >= 60) {
      color = 'orange.400';
      label = 'Greed';
    } else if (value >= 45) {
      color = 'yellow.400';
      label = 'Neutral';
    } else if (value >= 25) {
      color = 'green.300';
      label = 'Fear';
    } else {
      color = 'green.500';
      label = 'Extreme Fear';
    }

    return (
      <Box>
        <Text color={color} fontWeight="bold">
          {value} - {label}
        </Text>
        <Progress 
          value={value} 
          colorScheme={
            value >= 75 ? 'red' : 
            value >= 60 ? 'orange' : 
            value >= 45 ? 'yellow' : 
            value >= 25 ? 'green' : 'green'
          } 
          size="sm" 
          mt={2}
        />
      </Box>
    );
  };

  return (
    <Box p={5}>
      <Heading mb={5}>Market Sentiment Analysis</Heading>
      
      <Box mb={5}>
        <SimpleGrid columns={{ base: 1, md: 2 }} spacing={5} mb={5}>
          <Box>
            <Text mb={2}>Select Cryptocurrency</Text>
            <Select
              value={selectedCrypto}
              onChange={(e) => setSelectedCrypto(e.target.value)}
              width={{ base: 'full', md: '60%' }}
            >
              <option value="BTC">Bitcoin (BTC)</option>
              <option value="ETH">Ethereum (ETH)</option>
              <option value="BNB">Binance Coin (BNB)</option>
              <option value="XRP">Ripple (XRP)</option>
              <option value="ADA">Cardano (ADA)</option>
            </Select>
          </Box>
          
          <Box textAlign={{ base: 'left', md: 'right' }}>
            <Button
              colorScheme="blue"
              onClick={refreshSentimentData}
              isLoading={isRefreshing}
              loadingText="Refreshing..."
              mr={2}
            >
              Refresh Data
            </Button>
            
            {lastUpdated && (
              <Text fontSize="sm" color="gray.500" mt={2}>
                Last updated: {lastUpdated.toLocaleString()}
              </Text>
            )}
          </Box>
        </SimpleGrid>
      </Box>
      
      {isLoading ? (
        <Box textAlign="center" py={10}>
          <Spinner size="xl" />
          <Text mt={4}>Loading sentiment data...</Text>
        </Box>
      ) : sentimentData ? (
        <Tabs variant="enclosed">
          <TabList>
            <Tab>Overview</Tab>
            <Tab>Social Media</Tab>
            <Tab>News Analysis</Tab>
            <Tab>Historical Data</Tab>
          </TabList>
          
          <TabPanels>
            {/* Overview Tab */}
            <TabPanel>
              <SimpleGrid columns={{ base: 1, md: 3 }} spacing={5}>
                <Card>
                  <CardHeader>
                    <Heading size="md">Social Sentiment</Heading>
                  </CardHeader>
                  <CardBody>
                    <Stat>
                      <StatLabel>Overall Social Sentiment</StatLabel>
                      <StatNumber>
                        {renderSentimentScore(sentimentData.social_sentiment.overall)}
                      </StatNumber>
                      <StatHelpText>
                        <StatArrow 
                          type={sentimentData.social_sentiment.change_24h >= 0 ? 'increase' : 'decrease'} 
                        />
                        {Math.abs(sentimentData.social_sentiment.change_24h * 100).toFixed(1)}% in 24h
                      </StatHelpText>
                    </Stat>
                  </CardBody>
                </Card>
                
                <Card>
                  <CardHeader>
                    <Heading size="md">News Sentiment</Heading>
                  </CardHeader>
                  <CardBody>
                    <Stat>
                      <StatLabel>Overall News Sentiment</StatLabel>
                      <StatNumber>
                        {renderSentimentScore(sentimentData.news_sentiment.overall)}
                      </StatNumber>
                      <StatHelpText>
                        <StatArrow 
                          type={sentimentData.news_sentiment.change_24h >= 0 ? 'increase' : 'decrease'} 
                        />
                        {Math.abs(sentimentData.news_sentiment.change_24h * 100).toFixed(1)}% in 24h
                      </StatHelpText>
                    </Stat>
                  </CardBody>
                </Card>
                
                <Card>
                  <CardHeader>
                    <Heading size="md">Fear & Greed Index</Heading>
                  </CardHeader>
                  <CardBody>
                    <Stat>
                      <StatLabel>Current Index</StatLabel>
                      <StatNumber>
                        {renderFearGreedIndex(sentimentData.fear_greed_index.value)}
                      </StatNumber>
                      <StatHelpText>
                        <StatArrow 
                          type={sentimentData.fear_greed_index.change_24h >= 0 ? 'increase' : 'decrease'} 
                        />
                        {Math.abs(sentimentData.fear_greed_index.change_24h)} points in 24h
                      </StatHelpText>
                    </Stat>
                  </CardBody>
                </Card>
              </SimpleGrid>
              
              <Card mt={5}>
                <CardHeader>
                  <Heading size="md">Market Indicators</Heading>
                </CardHeader>
                <CardBody>
                  <SimpleGrid columns={{ base: 1, md: 3 }} spacing={5}>
                    <Stat>
                      <StatLabel>Volume Change (24h)</StatLabel>
                      <StatNumber>
                        <StatArrow 
                          type={sentimentData.market_indicators.volume_change >= 0 ? 'increase' : 'decrease'} 
                        />
                        {Math.abs(sentimentData.market_indicators.volume_change * 100).toFixed(1)}%
                      </StatNumber>
                    </Stat>
                    
                    <Stat>
                      <StatLabel>Volatility</StatLabel>
                      <StatNumber>
                        {(sentimentData.market_indicators.volatility * 100).toFixed(1)}%
                      </StatNumber>
                    </Stat>
                    
                    <Stat>
                      <StatLabel>Momentum</StatLabel>
                      <StatNumber>
                        {renderSentimentScore(sentimentData.market_indicators.momentum)}
                      </StatNumber>
                    </Stat>
                  </SimpleGrid>
                </CardBody>
              </Card>
              
              <Alert status="info" mt={5}>
                <AlertIcon />
                Sentiment analysis is now being used in price predictions. Higher sentiment scores generally correlate with positive price movements.
              </Alert>
            </TabPanel>
            
            {/* Social Media Tab */}
            <TabPanel>
              <Card>
                <CardHeader>
                  <Heading size="md">Social Media Sentiment Breakdown</Heading>
                </CardHeader>
                <CardBody>
                  <SimpleGrid columns={{ base: 1, md: 2 }} spacing={5}>
                    <Box>
                      <Text fontWeight="bold" mb={2}>Twitter Sentiment</Text>
                      <Progress 
                        value={sentimentData.social_sentiment.twitter * 100} 
                        colorScheme="twitter" 
                        size="lg" 
                        mb={4}
                      />
                      <Text>{(sentimentData.social_sentiment.twitter * 100).toFixed(1)}% positive</Text>
                    </Box>
                    
                    <Box>
                      <Text fontWeight="bold" mb={2}>Reddit Sentiment</Text>
                      <Progress 
                        value={sentimentData.social_sentiment.reddit * 100} 
                        colorScheme="orange" 
                        size="lg" 
                        mb={4}
                      />
                      <Text>{(sentimentData.social_sentiment.reddit * 100).toFixed(1)}% positive</Text>
                    </Box>
                  </SimpleGrid>
                  
                  <Box mt={5}>
                    <Text fontWeight="bold" mb={2}>Overall Social Sentiment</Text>
                    <Progress 
                      value={sentimentData.social_sentiment.overall * 100} 
                      colorScheme="blue" 
                      size="lg" 
                      mb={4}
                    />
                    <Text>{(sentimentData.social_sentiment.overall * 100).toFixed(1)}% positive</Text>
                  </Box>
                </CardBody>
              </Card>
              
              <Alert status="info" mt={5}>
                <AlertIcon />
                Social sentiment is calculated by analyzing thousands of social media posts and comments about {selectedCrypto}.
              </Alert>
            </TabPanel>
            
            {/* News Analysis Tab */}
            <TabPanel>
              <Card>
                <CardHeader>
                  <Heading size="md">News Sentiment Analysis</Heading>
                </CardHeader>
                <CardBody>
                  <SimpleGrid columns={{ base: 1, md: 3 }} spacing={5} mb={5}>
                    <Stat>
                      <StatLabel>Positive Articles</StatLabel>
                      <StatNumber color="green.500">
                        {sentimentData.news_sentiment.positive_articles}
                      </StatNumber>
                    </Stat>
                    
                    <Stat>
                      <StatLabel>Neutral Articles</StatLabel>
                      <StatNumber color="gray.500">
                        {sentimentData.news_sentiment.neutral_articles}
                      </StatNumber>
                    </Stat>
                    
                    <Stat>
                      <StatLabel>Negative Articles</StatLabel>
                      <StatNumber color="red.500">
                        {sentimentData.news_sentiment.negative_articles}
                      </StatNumber>
                    </Stat>
                  </SimpleGrid>
                  
                  <Box>
                    <Text fontWeight="bold" mb={2}>Overall News Sentiment</Text>
                    <Progress 
                      value={sentimentData.news_sentiment.overall * 100} 
                      colorScheme="green" 
                      size="lg" 
                      mb={4}
                    />
                    <Text>{(sentimentData.news_sentiment.overall * 100).toFixed(1)}% positive</Text>
                  </Box>
                </CardBody>
              </Card>
              
              <Alert status="info" mt={5}>
                <AlertIcon />
                News sentiment is calculated by analyzing recent news articles about {selectedCrypto} from major financial and crypto news sources.
              </Alert>
            </TabPanel>
            
            {/* Historical Data Tab */}
            <TabPanel>
              <Card>
                <CardHeader>
                  <Heading size="md">Historical Sentiment Data</Heading>
                </CardHeader>
                <CardBody>
                  <Table variant="simple">
                    <Thead>
                      <Tr>
                        <Th>Date</Th>
                        <Th>Social Sentiment</Th>
                        <Th>News Sentiment</Th>
                        <Th>Fear & Greed Index</Th>
                      </Tr>
                    </Thead>
                    <Tbody>
                      {sentimentData.historical.map((item) => (
                        <Tr key={item.date}>
                          <Td>{item.date}</Td>
                          <Td>{renderSentimentScore(item.social)}</Td>
                          <Td>{renderSentimentScore(item.news)}</Td>
                          <Td>
                            <Badge 
                              colorScheme={
                                item.fear_greed >= 75 ? 'red' : 
                                item.fear_greed >= 60 ? 'orange' : 
                                item.fear_greed >= 45 ? 'yellow' : 
                                item.fear_greed >= 25 ? 'green' : 'green'
                              }
                            >
                              {item.fear_greed}
                            </Badge>
                          </Td>
                        </Tr>
                      ))}
                    </Tbody>
                  </Table>
                </CardBody>
              </Card>
            </TabPanel>
          </TabPanels>
        </Tabs>
      ) : (
        <Alert status="warning">
          <AlertIcon />
          No sentiment data available for {selectedCrypto}. Please try refreshing or select a different cryptocurrency.
        </Alert>
      )}
    </Box>
  );
};

export default MarketSentiment;
