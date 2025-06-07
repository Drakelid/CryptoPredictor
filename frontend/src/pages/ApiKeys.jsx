import React from 'react';
import { Box, Heading, Text, List, ListItem } from '@chakra-ui/react';

const ApiKeys = () => (
  <Box p={5}>
    <Heading mb={5}>API Keys</Heading>
    <Text mb={3}>
      Configure these keys as environment variables for the backend. They are optional but enable higher rate limits and additional sentiment data.
    </Text>
    <List spacing={2} styleType="disc" pl={4}>
      <ListItem><strong>CoinGecko API Key</strong></ListItem>
      <ListItem><strong>Binance API Key</strong> and <strong>Secret</strong></ListItem>
      <ListItem><strong>CryptoPanic API Key</strong> for news sentiment</ListItem>
      <ListItem>
        <strong>Reddit Client ID</strong> and <strong>Client Secret</strong> (<code>REDDIT_CLIENT_ID</code>, <code>REDDIT_CLIENT_SECRET</code>)
        , along with optional <code>REDDIT_USER_AGENT</code>
      </ListItem>
    </List>
  </Box>
);

export default ApiKeys;
