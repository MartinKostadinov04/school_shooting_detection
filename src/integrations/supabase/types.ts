export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instantiate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "14.5"
  }
  public: {
    Tables: {
      department_schools: {
        Row: {
          department_id: string
          school_id: string
        }
        Insert: {
          department_id: string
          school_id: string
        }
        Update: {
          department_id?: string
          school_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "department_schools_department_id_fkey"
            columns: ["department_id"]
            isOneToOne: false
            referencedRelation: "police_departments"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "department_schools_school_id_fkey"
            columns: ["school_id"]
            isOneToOne: false
            referencedRelation: "schools"
            referencedColumns: ["id"]
          },
        ]
      }
      devices: {
        Row: {
          feed_url: string | null
          id: string
          last_event: string | null
          last_seen: string
          location: string
          name: string
          school_id: string
          status: Database["public"]["Enums"]["device_status"]
          type: Database["public"]["Enums"]["device_type"]
          x: number
          y: number
        }
        Insert: {
          feed_url?: string | null
          id?: string
          last_event?: string | null
          last_seen?: string
          location: string
          name: string
          school_id: string
          status?: Database["public"]["Enums"]["device_status"]
          type: Database["public"]["Enums"]["device_type"]
          x: number
          y: number
        }
        Update: {
          feed_url?: string | null
          id?: string
          last_event?: string | null
          last_seen?: string
          location?: string
          name?: string
          school_id?: string
          status?: Database["public"]["Enums"]["device_status"]
          type?: Database["public"]["Enums"]["device_type"]
          x?: number
          y?: number
        }
        Relationships: [
          {
            foreignKeyName: "devices_school_id_fkey"
            columns: ["school_id"]
            isOneToOne: false
            referencedRelation: "schools"
            referencedColumns: ["id"]
          },
        ]
      }
      incident_events: {
        Row: {
          actor_id: string | null
          detail: string | null
          id: string
          incident_id: string
          label: string
          timestamp: string
        }
        Insert: {
          actor_id?: string | null
          detail?: string | null
          id?: string
          incident_id: string
          label: string
          timestamp?: string
        }
        Update: {
          actor_id?: string | null
          detail?: string | null
          id?: string
          incident_id?: string
          label?: string
          timestamp?: string
        }
        Relationships: [
          {
            foreignKeyName: "incident_events_incident_id_fkey"
            columns: ["incident_id"]
            isOneToOne: false
            referencedRelation: "incidents"
            referencedColumns: ["id"]
          },
        ]
      }
      incidents: {
        Row: {
          audio_path: string | null
          created_at: string
          description: string | null
          display_id: string | null
          id: string
          location: string
          probability: number | null
          reported_by: string | null
          school_id: string
          severity: Database["public"]["Enums"]["incident_severity"]
          source: Database["public"]["Enums"]["incident_source"]
          status: Database["public"]["Enums"]["incident_status"]
          type: Database["public"]["Enums"]["incident_type"]
          video_confirmed: boolean
          video_path: string | null
        }
        Insert: {
          audio_path?: string | null
          created_at?: string
          description?: string | null
          display_id?: string | null
          id?: string
          location: string
          probability?: number | null
          reported_by?: string | null
          school_id: string
          severity: Database["public"]["Enums"]["incident_severity"]
          source: Database["public"]["Enums"]["incident_source"]
          status?: Database["public"]["Enums"]["incident_status"]
          type: Database["public"]["Enums"]["incident_type"]
          video_confirmed?: boolean
          video_path?: string | null
        }
        Update: {
          audio_path?: string | null
          created_at?: string
          description?: string | null
          display_id?: string | null
          id?: string
          location?: string
          probability?: number | null
          reported_by?: string | null
          school_id?: string
          severity?: Database["public"]["Enums"]["incident_severity"]
          source?: Database["public"]["Enums"]["incident_source"]
          status?: Database["public"]["Enums"]["incident_status"]
          type?: Database["public"]["Enums"]["incident_type"]
          video_confirmed?: boolean
          video_path?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "incidents_school_id_fkey"
            columns: ["school_id"]
            isOneToOne: false
            referencedRelation: "schools"
            referencedColumns: ["id"]
          },
        ]
      }
      messages: {
        Row: {
          id: string
          incident_id: string | null
          incident_report: Json | null
          school_id: string
          sender_id: string
          sender_role: Database["public"]["Enums"]["app_role"]
          text: string | null
          timestamp: string
        }
        Insert: {
          id?: string
          incident_id?: string | null
          incident_report?: Json | null
          school_id: string
          sender_id: string
          sender_role: Database["public"]["Enums"]["app_role"]
          text?: string | null
          timestamp?: string
        }
        Update: {
          id?: string
          incident_id?: string | null
          incident_report?: Json | null
          school_id?: string
          sender_id?: string
          sender_role?: Database["public"]["Enums"]["app_role"]
          text?: string | null
          timestamp?: string
        }
        Relationships: [
          {
            foreignKeyName: "messages_incident_id_fkey"
            columns: ["incident_id"]
            isOneToOne: false
            referencedRelation: "incidents"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "messages_school_id_fkey"
            columns: ["school_id"]
            isOneToOne: false
            referencedRelation: "schools"
            referencedColumns: ["id"]
          },
        ]
      }
      notifications: {
        Row: {
          audio_path: string | null
          device_type: Database["public"]["Enums"]["device_type"]
          id: string
          incident_id: string | null
          location: string
          message: string
          probability: number | null
          school_id: string
          source: Database["public"]["Enums"]["incident_source"]
          timestamp: string
          video_path: string | null
        }
        Insert: {
          audio_path?: string | null
          device_type: Database["public"]["Enums"]["device_type"]
          id?: string
          incident_id?: string | null
          location: string
          message: string
          probability?: number | null
          school_id: string
          source: Database["public"]["Enums"]["incident_source"]
          timestamp?: string
          video_path?: string | null
        }
        Update: {
          audio_path?: string | null
          device_type?: Database["public"]["Enums"]["device_type"]
          id?: string
          incident_id?: string | null
          location?: string
          message?: string
          probability?: number | null
          school_id?: string
          source?: Database["public"]["Enums"]["incident_source"]
          timestamp?: string
          video_path?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "notifications_incident_id_fkey"
            columns: ["incident_id"]
            isOneToOne: false
            referencedRelation: "incidents"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "notifications_school_id_fkey"
            columns: ["school_id"]
            isOneToOne: false
            referencedRelation: "schools"
            referencedColumns: ["id"]
          },
        ]
      }
      police_departments: {
        Row: {
          created_at: string
          id: string
          jurisdiction: string
          name: string
        }
        Insert: {
          created_at?: string
          id?: string
          jurisdiction: string
          name: string
        }
        Update: {
          created_at?: string
          id?: string
          jurisdiction?: string
          name?: string
        }
        Relationships: []
      }
      schools: {
        Row: {
          address: string
          created_at: string
          district: string
          floorplan_path: string | null
          id: string
          name: string
          student_count: number
        }
        Insert: {
          address: string
          created_at?: string
          district: string
          floorplan_path?: string | null
          id?: string
          name: string
          student_count?: number
        }
        Update: {
          address?: string
          created_at?: string
          district?: string
          floorplan_path?: string | null
          id?: string
          name?: string
          student_count?: number
        }
        Relationships: []
      }
      user_roles: {
        Row: {
          created_at: string
          department_id: string | null
          id: string
          role: Database["public"]["Enums"]["app_role"]
          school_id: string | null
          user_id: string
        }
        Insert: {
          created_at?: string
          department_id?: string | null
          id?: string
          role: Database["public"]["Enums"]["app_role"]
          school_id?: string | null
          user_id: string
        }
        Update: {
          created_at?: string
          department_id?: string | null
          id?: string
          role?: Database["public"]["Enums"]["app_role"]
          school_id?: string | null
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "user_roles_department_id_fkey"
            columns: ["department_id"]
            isOneToOne: false
            referencedRelation: "police_departments"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "user_roles_school_id_fkey"
            columns: ["school_id"]
            isOneToOne: false
            referencedRelation: "schools"
            referencedColumns: ["id"]
          },
        ]
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      has_department_access: { Args: { _department: string }; Returns: boolean }
      has_role: {
        Args: {
          _role: Database["public"]["Enums"]["app_role"]
          _user_id: string
        }
        Returns: boolean
      }
      has_school_access: { Args: { _school: string }; Returns: boolean }
    }
    Enums: {
      app_role: "school_operator" | "police_dispatcher" | "admin"
      device_status: "online" | "warning" | "triggered" | "offline"
      device_type: "camera" | "microphone"
      incident_severity: "Low" | "Medium" | "High" | "Critical"
      incident_source: "AUDIO-AI" | "VIDEO-AI" | "MANUAL"
      incident_status: "NEW" | "ACKNOWLEDGED" | "RESOLVED"
      incident_type:
        | "Gunshot"
        | "Suspicious Activity"
        | "Fire"
        | "Medical"
        | "Other"
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {
      app_role: ["school_operator", "police_dispatcher", "admin"],
      device_status: ["online", "warning", "triggered", "offline"],
      device_type: ["camera", "microphone"],
      incident_severity: ["Low", "Medium", "High", "Critical"],
      incident_source: ["AUDIO-AI", "VIDEO-AI", "MANUAL"],
      incident_status: ["NEW", "ACKNOWLEDGED", "RESOLVED"],
      incident_type: [
        "Gunshot",
        "Suspicious Activity",
        "Fire",
        "Medical",
        "Other",
      ],
    },
  },
} as const
