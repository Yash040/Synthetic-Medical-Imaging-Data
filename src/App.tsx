import React, { useState } from 'react';
import { Brain, Settings as Lungs, Spline as Spine } from 'lucide-react';

type ScanType = 'xray' | 'ct' | 'mri';
type Disease = 'chest' | 'brain' | 'abdomen';

interface FormData {
  patientName: string;
  age: string;
  scanType: ScanType;
  disease: Disease;
}

function App() {
  const [formData, setFormData] = useState<FormData>({
    patientName: '',
    age: '',
    scanType: 'xray',
    disease: 'chest'
  });

  const [generatedImage, setGeneratedImage] = useState<string | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    // Simulate AI generation with sample images based on selection
    const images = {
      xray: "https://images.unsplash.com/photo-1516069677018-378ddb5ea25f?auto=format&fit=crop&w=800",
      ct: "https://images.unsplash.com/photo-1583912267550-d6c2ac4b0154?auto=format&fit=crop&w=800",
      mri: "https://images.unsplash.com/photo-1583911860205-72f8de576c82?auto=format&fit=crop&w=800"
    };

    setGeneratedImage(images[formData.scanType]);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value,
      disease: name === 'scanType' 
        ? value === 'xray' 
          ? 'chest' 
          : value === 'ct' 
            ? 'brain' 
            : 'abdomen'
        : prev.disease
    }));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Healthcare Imaging Synthetic Data Generator</h1>
          <p className="text-gray-600">Generate synthetic medical images using AI</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-white p-6 rounded-xl shadow-lg">
            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Patient Name
                </label>
                <input
                  type="text"
                  name="patientName"
                  value={formData.patientName}
                  onChange={handleInputChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Age
                </label>
                <input
                  type="number"
                  name="age"
                  value={formData.age}
                  onChange={handleInputChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Scan Type
                </label>
                <select
                  name="scanType"
                  value={formData.scanType}
                  onChange={handleInputChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="xray">X-Ray (Chest)</option>
                  <option value="ct">CT Scan (Brain)</option>
                  <option value="mri">MRI (Abdomen)</option>
                </select>
              </div>

              <button
                type="submit"
                className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition duration-200"
              >
                Generate Image
              </button>
            </form>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-lg">
            <h2 className="text-xl font-semibold mb-4">Generated Image</h2>
            <div className="aspect-square bg-gray-100 rounded-lg overflow-hidden">
              {generatedImage ? (
                <img
                  src={generatedImage}
                  alt="Generated medical scan"
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <div className="text-center text-gray-500">
                    {formData.scanType === 'xray' && <Lungs size={48} className="mx-auto mb-2" />}
                    {formData.scanType === 'ct' && <Brain size={48} className="mx-auto mb-2" />}
                    {formData.scanType === 'mri' && <Spine size={48} className="mx-auto mb-2" />}
                    <p>Generated image will appear here</p>
                  </div>
                </div>
              )}
            </div>
            {generatedImage && (
              <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                <h3 className="font-medium text-gray-900">Image Details</h3>
                <p className="text-sm text-gray-600">Patient: {formData.patientName}</p>
                <p className="text-sm text-gray-600">Age: {formData.age}</p>
                <p className="text-sm text-gray-600">
                  Type: {formData.scanType.toUpperCase()} - {formData.disease}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;