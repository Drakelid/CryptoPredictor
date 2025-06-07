import React, { useState, useEffect } from 'react';
import {
  Box,
  Heading,
  Stack,
  FormControl,
  FormLabel,
  Input,
  Button,
  useToast,
} from '@chakra-ui/react';

const ApiKeys = () => {
  const [keys, setKeys] = useState({
    coingecko: '',
    binanceKey: '',
    binanceSecret: '',
    cryptopanic: '',
    redditClientId: '',
    redditClientSecret: '',
    redditUserAgent: '',
  });

  const toast = useToast();

  useEffect(() => {
    const stored = localStorage.getItem('apiKeys');
    if (stored) {
      setKeys(JSON.parse(stored));
    }
  }, []);

  const handleChange = (e) => {
    setKeys({ ...keys, [e.target.name]: e.target.value });
  };

  const handleSave = () => {
    localStorage.setItem('apiKeys', JSON.stringify(keys));
    toast({
      title: 'Saved',
      status: 'success',
      duration: 3000,
      isClosable: true,
    });
  };

  return (
    <Box p={5} maxW="md">
      <Heading mb={5}>API Keys</Heading>
      <Stack spacing={4}>
        <FormControl>
          <FormLabel>CoinGecko API Key</FormLabel>
          <Input
            placeholder="COINGECKO_API_KEY"
            name="coingecko"
            value={keys.coingecko}
            onChange={handleChange}
          />
        </FormControl>
        <FormControl>
          <FormLabel>Binance API Key</FormLabel>
          <Input
            placeholder="BINANCE_API_KEY"
            name="binanceKey"
            value={keys.binanceKey}
            onChange={handleChange}
          />
        </FormControl>
        <FormControl>
          <FormLabel>Binance API Secret</FormLabel>
          <Input
            placeholder="BINANCE_API_SECRET"
            name="binanceSecret"
            value={keys.binanceSecret}
            onChange={handleChange}
          />
        </FormControl>
        <FormControl>
          <FormLabel>CryptoPanic API Key</FormLabel>
          <Input
            placeholder="NEWS_API_KEY"
            name="cryptopanic"
            value={keys.cryptopanic}
            onChange={handleChange}
          />
        </FormControl>
        <FormControl>
          <FormLabel>Reddit Client ID</FormLabel>
          <Input
            placeholder="REDDIT_CLIENT_ID"
            name="redditClientId"
            value={keys.redditClientId}
            onChange={handleChange}
          />
        </FormControl>
        <FormControl>
          <FormLabel>Reddit Client Secret</FormLabel>
          <Input
            placeholder="REDDIT_CLIENT_SECRET"
            name="redditClientSecret"
            value={keys.redditClientSecret}
            onChange={handleChange}
          />
        </FormControl>
        <FormControl>
          <FormLabel>Reddit User Agent</FormLabel>
          <Input
            placeholder="REDDIT_USER_AGENT"
            name="redditUserAgent"
            value={keys.redditUserAgent}
            onChange={handleChange}
          />
        </FormControl>
        <Button colorScheme="blue" onClick={handleSave} width="full">
          Save Locally
        </Button>
      </Stack>
    </Box>
  );
};

export default ApiKeys;
