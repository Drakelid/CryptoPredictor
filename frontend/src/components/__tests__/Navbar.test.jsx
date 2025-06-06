import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ChakraProvider } from '@chakra-ui/react'
import { BrowserRouter as Router } from 'react-router-dom'
import Navbar from '../Navbar'

describe('Navbar', () => {
  it('renders application title', () => {
    window.matchMedia = window.matchMedia || function() {
      return {
        matches: false,
        addListener: () => {},
        removeListener: () => {}
      }
    }
    render(
      <ChakraProvider>
        <Router>
          <Navbar />
        </Router>
      </ChakraProvider>
    )
    expect(screen.getByText(/CryptoPricer/i)).not.toBeNull()
  })
})
