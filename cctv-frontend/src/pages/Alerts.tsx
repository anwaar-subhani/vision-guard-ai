import { Card, CardContent } from '../components/ui/card'
import { Badge } from '../components/ui/badge'
import { Button } from '../components/ui/button'
import { 
  AlertTriangle, 
  Clock, 
  FileVideo, 
  Filter,
  Download,
  Eye,
  Trash2,
  Volume2,
  Activity,
  Shield
} from 'lucide-react'

export default function Alerts() {
  const alerts = [
    {
      id: 1,
      type: 'Suspicious Activity',
      description: 'Unusual movement pattern detected in uploaded video',
      filename: 'security_footage_001.mp4',
      timestamp: '2 min ago',
      severity: 'high',
      status: 'active',
      confidence: 94.5,
      videoTime: '1:23'
    },
    {
      id: 2,
      type: 'Gunshot Sound',
      description: 'Gunshot sound detected in audio analysis',
      filename: 'parking_lot_night.mp4',
      timestamp: '5 min ago',
      severity: 'critical',
      status: 'investigating',
      confidence: 98.2,
      videoTime: '2:45'
    },
    {
      id: 3,
      type: 'Crowd Gathering',
      description: 'Large group formation detected in video',
      filename: 'lobby_morning.mp4',
      timestamp: '8 min ago',
      severity: 'medium',
      status: 'resolved',
      confidence: 87.3,
      videoTime: '3:20'
    },
    {
      id: 4,
      type: 'Medical Emergency',
      description: 'Person appears to have fainted or fallen',
      filename: 'hallway_cam.mp4',
      timestamp: '12 min ago',
      severity: 'critical',
      status: 'resolved',
      confidence: 91.7,
      videoTime: '4:15'
    },
    {
      id: 5,
      type: 'Unauthorized Access',
      description: 'Person detected in restricted area',
      filename: 'server_room.mp4',
      timestamp: '15 min ago',
      severity: 'high',
      status: 'active',
      confidence: 96.1,
      videoTime: '0:45'
    }
  ]

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'destructive'
      case 'high': return 'default'
      case 'medium': return 'secondary'
      case 'low': return 'outline'
      default: return 'outline'
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'destructive'
      case 'investigating': return 'default'
      case 'resolved': return 'secondary'
      default: return 'outline'
    }
  }

  const getSeverityBg = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-50 border-red-200'
      case 'high': return 'bg-orange-50 border-orange-200'
      case 'medium': return 'bg-yellow-50 border-yellow-200'
      case 'low': return 'bg-blue-50 border-blue-200'
      default: return 'bg-gray-50 border-gray-200'
    }
  }

  const getTypeIcon = (type: string) => {
    if (type.includes('Sound') || type.includes('Audio')) {
      return Volume2
    }
    return AlertTriangle
  }

  return (
    <div className="space-y-6">
      <div className="px-4 sm:px-0">
        <h2 className="text-xl sm:text-2xl font-bold text-gray-900">Security Alerts</h2>
        <p className="text-sm sm:text-base text-gray-600 mt-1">
          AI-detected anomalies from your video analysis
        </p>
      </div>

      {/* Alert Summary */}
      <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
        <Card className="bg-white border border-gray-200 shadow-sm">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Alerts</p>
                <p className="text-2xl font-bold text-gray-900">{alerts.length}</p>
              </div>
              <div className="p-2 rounded-lg bg-blue-50">
                <AlertTriangle className="h-5 w-5 text-blue-600" />
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="bg-white border border-gray-200 shadow-sm">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Active Alerts</p>
                <p className="text-2xl font-bold text-red-600">
                  {alerts.filter(alert => alert.status === 'active').length}
                </p>
              </div>
              <div className="p-2 rounded-lg bg-red-50">
                <Activity className="h-5 w-5 text-red-600" />
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="bg-white border border-gray-200 shadow-sm">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Critical</p>
                <p className="text-2xl font-bold text-red-600">
                  {alerts.filter(alert => alert.severity === 'critical').length}
                </p>
              </div>
              <div className="p-2 rounded-lg bg-red-50">
                <AlertTriangle className="h-5 w-5 text-red-600" />
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="bg-white border border-gray-200 shadow-sm">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Resolved</p>
                <p className="text-2xl font-bold text-green-600">
                  {alerts.filter(alert => alert.status === 'resolved').length}
                </p>
              </div>
              <div className="p-2 rounded-lg bg-green-50">
                <Shield className="h-5 w-5 text-green-600" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filter and Export */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" className="border-gray-300">
            <Filter className="h-4 w-4 mr-2" />
            Filter
          </Button>
          <Button variant="outline" size="sm" className="border-gray-300">
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* Alerts List */}
      <div className="space-y-4">
        {alerts.map((alert) => {
          const TypeIcon = getTypeIcon(alert.type)
          return (
            <Card key={alert.id} className={`bg-white border shadow-sm ${getSeverityBg(alert.severity)}`}>
              <CardContent className="p-4">
                <div className="flex items-start justify-between">
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <TypeIcon className="h-4 w-4 text-gray-600" />
                      <h3 className="font-semibold text-gray-900">{alert.type}</h3>
                      <Badge 
                        variant={getSeverityColor(alert.severity)}
                        className="text-xs"
                      >
                        {alert.severity}
                      </Badge>
                      <Badge 
                        variant={getStatusColor(alert.status)}
                        className="text-xs"
                      >
                        {alert.status}
                      </Badge>
                    </div>
                    <p className="text-sm text-gray-700">{alert.description}</p>
                    <div className="flex items-center gap-4 text-sm text-gray-600">
                      <div className="flex items-center gap-1">
                        <FileVideo className="h-3 w-3" />
                        {alert.filename}
                      </div>
                      <div className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {alert.timestamp}
                      </div>
                      <div className="text-gray-500">
                        Video Time: {alert.videoTime}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium text-gray-900">Confidence: {alert.confidence}%</div>
                    <div className="flex items-center gap-2 mt-2">
                      <Button variant="outline" size="sm" className="border-gray-300">
                        <Eye className="h-3 w-3 mr-1" />
                        View
                      </Button>
                      <Button variant="outline" size="sm" className="border-gray-300">
                        <Trash2 className="h-3 w-3 mr-1" />
                        Dismiss
                      </Button>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>
    </div>
  )
}
