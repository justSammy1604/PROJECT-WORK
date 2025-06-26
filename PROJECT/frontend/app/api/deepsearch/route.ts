import { NextResponse } from 'next/server'

export async function POST(request: Request) {
  try {
    const { query, history } = await request.json()
    console.log('Sending to backend:', { query })
    const response = await fetch(`http://localhost:8000/deepsearch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, history }),
    })
    console.log('Backend response:', response.status, response.statusText)
    if (!response.ok) {
      throw new Error(`Backend error: ${response.statusText}`)
    }
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('API error:', error)
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 })
  }
}