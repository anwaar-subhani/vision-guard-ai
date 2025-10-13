import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import { Badge } from '../components/ui/badge'
import { Button } from '../components/ui/button'
import { 
  Upload, 
  AlertTriangle, 
  Activity, 
  Shield,
  TrendingUp,
  Clock,
  FileVideo,
  CheckCircle,
  BarChart3
} from 'lucide-react'

export default function Dashboard() {
  const stats = [
    { title: 'Videos Analyzed', value: '24', icon: FileVideo, color: 'text-blue-600', bg: 'bg-blue-50' },
    { title: 'Anomalies Detected', value: '8', icon: AlertTriangle, color: 'text-red-600', bg: 'bg-red-50' },
    { title: 'AI Accuracy', value: '98.5%', icon: Shield, color: 'text-green-600', bg: 'bg-green-50' },
    { title: 'Processing Time', value: '2.3s', icon: Activity, color: 'text-purple-600', bg: 'bg-purple-50' },
  ]

  const recentAnalyses = [
    { id: 1, filename: 'security_footage_001.mp4', status: 'completed', anomalies: 3, time: '2 min ago', confidence: 94 },
    { id: 2, filename: 'parking_lot_night.mp4', status: 'processing', anomalies: 0, time: '5 min ago', confidence: 0 },
    { id: 3, filename: 'lobby_morning.mp4', status: 'completed', anomalies: 1, time: '8 min ago', confidence: 87 },
  ]

  const quickActions = [
    { 
      title: 'Upload Video', 
      description: 'Upload CCTV footage for AI analysis', 
      icon: Upload, 
      href: '/analysis', 
      color: 'text-white', 
      bg: 'bg-[#7075b8] hover:bg-[#5a5f9a]',
      primary: true
    },
    { 
      title: 'View Analytics', 
      description: 'Check system performance metrics', 
      icon: BarChart3, 
      href: '/analytics', 
      color: 'text-gray-700', 
      bg: 'bg-gray-50 hover:bg-gray-100',
      primary: false
    },
  ]

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h2 className="text-3xl font-bold text-gray-900">Welcome to VisionGuard AI</h2>
        <p className="text-lg text-gray-600 mt-2">
          Upload your CCTV footage and let AI detect anomalies automatically
        </p>
      </div>

      {/* Stats Overview */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <Card key={stat.title} className="bg-white border border-gray-200 shadow-sm hover:shadow-md transition-shadow">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">{stat.title}</p>
                  <p className="text-2xl font-bold text-gray-900">{stat.value}</p>
                </div>
                <div className={`p-3 rounded-xl ${stat.bg}`}>
                  <stat.icon className={`h-6 w-6 ${stat.color}`} />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>


      <div className="grid gap-6 lg:grid-cols-2">
        {/* Recent Analyses */}
        <Card className="bg-white border border-gray-200 shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-gray-800">
              <FileVideo className="h-5 w-5" />
              Recent Analyses
            </CardTitle>
            <CardDescription>
              Latest video processing results
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {recentAnalyses.map((analysis) => (
                <div key={analysis.id} className="flex items-center justify-between p-3 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-gray-900">{analysis.filename}</span>
                      <Badge 
                        variant={analysis.status === 'completed' ? 'default' : 'secondary'}
                        className={analysis.status === 'completed' ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}
                      >
                        {analysis.status}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-4 text-sm text-gray-600">
                      <span>Anomalies: {analysis.anomalies}</span>
                      {analysis.confidence > 0 && (
                        <span>Confidence: {analysis.confidence}%</span>
                      )}
                    </div>
                    <div className="flex items-center gap-1 text-sm text-gray-500">
                      <Clock className="h-3 w-3" />
                      {analysis.time}
                    </div>
                  </div>
                  <Button variant="outline" size="sm">
                    View
                  </Button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Quick Actions */}
        <Card className="bg-white border border-gray-200 shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-gray-800">
              <Activity className="h-5 w-5" />
              Quick Actions
            </CardTitle>
            <CardDescription>
              Access system features
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {quickActions.map((action) => (
                <Button
                  key={action.title}
                  variant={action.primary ? "default" : "outline"}
                  className={`w-full h-auto p-4 justify-start ${action.bg} ${action.color} hover:shadow-md transition-all`}
                  onClick={() => window.location.href = action.href}
                >
                  <div className="flex items-center gap-3">
                    <action.icon className="h-5 w-5" />
                    <div className="text-left">
                      <div className="font-medium">{action.title}</div>
                      <div className="text-sm opacity-80">{action.description}</div>
                    </div>
                  </div>
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
