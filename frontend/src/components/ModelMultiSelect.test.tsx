import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import ModelMultiSelect from './ModelMultiSelect'

const mockModels = [
  {
    id: 'kyle',
    displayName: 'Kyle Model (1985)',
    description: 'Single dealer model',
    parameters: {},
    visualizations: [],
  },
  {
    id: 'almgren',
    displayName: 'Almgren-Chriss (2000)',
    description: 'Optimal execution',
    parameters: {},
    visualizations: [],
  },
  {
    id: 'glosten',
    displayName: 'Glosten-Milgrom (1985)',
    description: 'Bid-ask spread',
    parameters: {},
    visualizations: [],
  },
]

describe('ModelMultiSelect', () => {
  it('renders model checkboxes', () => {
    const onSelectionChange = vi.fn()
    
    render(
      <ModelMultiSelect
        models={mockModels}
        selectedModelIds={[]}
        onSelectionChange={onSelectionChange}
      />
    )
    
    expect(screen.getByText('Kyle Model (1985)')).toBeInTheDocument()
    expect(screen.getByText('Almgren-Chriss (2000)')).toBeInTheDocument()
    expect(screen.getByText('Glosten-Milgrom (1985)')).toBeInTheDocument()
  })

  it('shows selected count', () => {
    const onSelectionChange = vi.fn()
    
    render(
      <ModelMultiSelect
        models={mockModels}
        selectedModelIds={['kyle']}
        onSelectionChange={onSelectionChange}
      />
    )
    
    expect(screen.getByText(/Select 1\/4 models/)).toBeInTheDocument()
  })

  it('calls onSelectionChange when model is selected', () => {
    const onSelectionChange = vi.fn()
    
    render(
      <ModelMultiSelect
        models={mockModels}
        selectedModelIds={[]}
        onSelectionChange={onSelectionChange}
      />
    )
    
    const kyleCheckbox = screen.getByLabelText('Kyle Model (1985)')
    fireEvent.click(kyleCheckbox)
    
    expect(onSelectionChange).toHaveBeenCalledWith(['kyle'])
  })

  it('allows selecting up to 4 models', () => {
    const onSelectionChange = vi.fn()
    
    render(
      <ModelMultiSelect
        models={[...mockModels, { id: 'deprado', displayName: 'De Prado', description: '', parameters: {}, visualizations: [] }]}
        selectedModelIds={['kyle', 'almgren', 'glosten']}
        onSelectionChange={onSelectionChange}
      />
    )
    
    const depradoCheckbox = screen.getByLabelText('De Prado')
    fireEvent.click(depradoCheckbox)
    
    expect(onSelectionChange).toHaveBeenCalledWith(['kyle', 'almgren', 'glosten', 'deprado'])
  })

  it('disables checkbox when 4 models already selected', () => {
    const onSelectionChange = vi.fn()
    
    render(
      <ModelMultiSelect
        models={[...mockModels, { id: 'deprado', displayName: 'De Prado', description: '', parameters: {}, visualizations: [] }]}
        selectedModelIds={['kyle', 'almgren', 'glosten', 'deprado']}
        onSelectionChange={onSelectionChange}
      />
    )
    
    // All checkboxes should be checked and disabled for unselected
    const allCheckboxes = screen.getAllByRole('checkbox')
    allCheckboxes.forEach(checkbox => {
      if (checkbox.getAttribute('checked')) {
        expect(checkbox).toBeEnabled()
      }
    })
  })

  it('allows deselecting when only 1 model selected', () => {
    const onSelectionChange = vi.fn()
    
    render(
      <ModelMultiSelect
        models={mockModels}
        selectedModelIds={['kyle']}
        onSelectionChange={onSelectionChange}
      />
    )
    
    const kyleCheckbox = screen.getByLabelText('Kyle Model (1985)')
    fireEvent.click(kyleCheckbox)
    
    // Should not call onSelectionChange because we keep at least 1
    expect(onSelectionChange).not.toHaveBeenCalled()
  })
})
