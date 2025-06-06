import React, { useState, useEffect } from 'react';
import {
  Box,
  Heading,
  FormControl,
  FormLabel,
  Input,
  Button,
  useToast,
  Text,
} from '@chakra-ui/react';
import axios from 'axios';
import { API_BASE_URL } from '../config';

const ApiKeys = () => {
  const [coinmarketcapKey, setCoinmarketcapKey] = useState('');
  const [binanceKey, setBinanceKey] = useState('');
  const [binanceSecret, setBinanceSecret] = useState('');
  const [newsKey, setNewsKey] = useState('');
  const [status, setStatus] = useState({});
  const toast = useToast();

  const fetchStatus = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/api/keys`);
      setStatus(res.data);
    } catch (err) {
      console.error('Error fetching key status', err);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await axios.post(`${API_BASE_URL}/api/keys`, {
        coinmarketcap_api_key: coinmarketcapKey || null,
        binance_api_key: binanceKey || null,
        binance_api_secret: binanceSecret || null,
        news_api_key: newsKey || null,
      });
      toast({ title: 'API keys saved', status: 'success', duration: 3000, isClosable: true });
      setCoinmarketcapKey('');
      setBinanceKey('');
      setBinanceSecret('');
      setNewsKey('');
      fetchStatus();
    } catch (err) {
      console.error('Error saving API keys', err);
      toast({ title: 'Failed to save API keys', status: 'error', duration: 3000, isClosable: true });
    }
  };

  return (
    <Box maxW="md" mx="auto">
      <Heading size="md" mb={4}>Manage API Keys</Heading>
      <form onSubmit={handleSubmit}>
        <FormControl mb={3}>
          <FormLabel>CoinMarketCap API Key</FormLabel>
          <Input value={coinmarketcapKey} onChange={(e) => setCoinmarketcapKey(e.target.value)} />
          {status.coinmarketcap_api_key ? <Text fontSize="sm" color="green.600">Key set</Text> : null}
        </FormControl>
        <FormControl mb={3}>
          <FormLabel>Binance API Key</FormLabel>
          <Input value={binanceKey} onChange={(e) => setBinanceKey(e.target.value)} />
          {status.binance_api_key ? <Text fontSize="sm" color="green.600">Key set</Text> : null}
        </FormControl>
        <FormControl mb={3}>
          <FormLabel>Binance API Secret</FormLabel>
          <Input value={binanceSecret} onChange={(e) => setBinanceSecret(e.target.value)} />
          {status.binance_api_secret ? <Text fontSize="sm" color="green.600">Secret set</Text> : null}
        </FormControl>
        <FormControl mb={3}>
          <FormLabel>News API Key</FormLabel>
          <Input value={newsKey} onChange={(e) => setNewsKey(e.target.value)} />
          {status.news_api_key ? <Text fontSize="sm" color="green.600">Key set</Text> : null}
        </FormControl>
        <Button type="submit" colorScheme="blue" mt={2}>Save Keys</Button>
      </form>
    </Box>
  );
};

export default ApiKeys;
