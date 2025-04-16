import React from 'react';
import {
  Box,
  Text,
  Heading,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Progress,
  SimpleGrid,
  Card,
  CardBody,
  CardHeader,
} from '@chakra-ui/react';

const ExplanationView = ({ explanation }) => {
  if (!explanation || !explanation.feature_importance) {
    return <Box>No explanation data available</Box>;
  }

  // Sort features by importance
  const sortedFeatures = [...explanation.feature_importance]
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 10); // Show top 10 features

  // Find max importance for scaling
  const maxImportance = Math.max(...sortedFeatures.map(f => f.importance));

  // Format feature names for display
  const formatFeatureName = (name) => {
    return name
      .replace(/_/g, ' ')
      .replace(/([A-Z])/g, ' $1')
      .replace(/^./, str => str.toUpperCase());
  };

  // Check if a feature is sentiment-related
  const isSentimentFeature = (name) => {
    const sentimentKeywords = ['sentiment', 'social', 'news', 'fear', 'greed', 'bullish', 'bearish', 'mention'];
    return sentimentKeywords.some(keyword => name.toLowerCase().includes(keyword));
  };

  return (
    <Box>
      <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
        <Card>
          <CardHeader>
            <Heading size="sm">Feature Importance</Heading>
          </CardHeader>
          <CardBody>
            <Table variant="simple" size="sm">
              <Thead>
                <Tr>
                  <Th>Feature</Th>
                  <Th>Importance</Th>
                  <Th>Relative Impact</Th>
                </Tr>
              </Thead>
              <Tbody>
                {sortedFeatures.map((feature, index) => (
                  <Tr key={index}>
                    <Td>
                      {formatFeatureName(feature.feature)}
                      {isSentimentFeature(feature.feature) && (
                        <Text as="span" ml={2} fontSize="xs" color="orange.500" fontWeight="bold">
                          (Sentiment)
                        </Text>
                      )}
                    </Td>
                    <Td isNumeric>{feature.importance.toFixed(4)}</Td>
                    <Td>
                      <Progress
                        value={(feature.importance / maxImportance) * 100}
                        size="sm"
                        colorScheme={isSentimentFeature(feature.feature) ? "orange" : "blue"}
                        borderRadius="full"
                      />
                    </Td>
                  </Tr>
                ))}
              </Tbody>
            </Table>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>
            <Heading size="sm">Interpretation</Heading>
          </CardHeader>
          <CardBody>
            <Text mb={4}>
              The model's prediction is primarily influenced by the following factors:
            </Text>
            <Box>
              {sortedFeatures.slice(0, 3).map((feature, index) => (
                <Text key={index} mb={2}>
                  <strong>{index + 1}. {formatFeatureName(feature.feature)}</strong>:
                  {' '}This feature accounts for {(feature.importance / maxImportance * 100).toFixed(1)}%
                  of the model's decision making process.
                </Text>
              ))}
            </Box>
            <Text mt={4} fontStyle="italic">
              Note: These explanations help understand which features the model considers most important
              when making predictions, but do not indicate causality.
            </Text>
          </CardBody>
        </Card>
      </SimpleGrid>

      {explanation.shap_values && (
        <Card mt={6}>
          <CardHeader>
            <Heading size="sm">SHAP Values</Heading>
          </CardHeader>
          <CardBody>
            <Text>
              SHAP values show how each feature contributes to pushing the prediction higher or lower.
              Positive values push the prediction higher, while negative values push it lower.
            </Text>
            {/* SHAP visualization would go here in a real implementation */}
            <Text mt={4} fontStyle="italic">
              A detailed SHAP visualization would be displayed here in the full implementation.
            </Text>
          </CardBody>
        </Card>
      )}

      <Card mt={6}>
        <CardHeader>
          <Heading size="sm">Market Sentiment Analysis</Heading>
        </CardHeader>
        <CardBody>
          <Text mb={4}>
            This prediction incorporates real-time market sentiment data from multiple sources:
          </Text>
          <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
            <Box>
              <Heading size="xs" mb={2}>Sentiment Sources</Heading>
              <Text>• Social Media Sentiment</Text>
              <Text>• News Sentiment Analysis</Text>
              <Text>• Fear & Greed Index</Text>
              <Text>• Trading Volume Analysis</Text>
            </Box>
            <Box>
              <Heading size="xs" mb={2}>Sentiment Impact</Heading>
              <Text>
                Market sentiment can significantly influence short-term price movements, often
                preceding changes in technical indicators. Our model combines sentiment signals
                with technical analysis for more accurate predictions.
              </Text>
            </Box>
          </SimpleGrid>
        </CardBody>
      </Card>
    </Box>
  );
};

export default ExplanationView;
