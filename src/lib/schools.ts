/**
 * Schools accessible to the current police department.
 *
 * TRANSFER: in production, fetch this list from the backend based on the
 * authenticated officer's department / jurisdiction
 *   GET /api/police/{departmentId}/schools  ->  School[]
 * Each school maps 1:1 to its own Ably channel + device inventory + incident
 * stream. The current demo store is single-tenant; when porting, scope the
 * incident store + Ably channel by `schoolId`.
 */

export interface School {
  id: string;
  name: string;
  address: string;
  district: string;
  /** Quick stats — TRANSFER: derive from real backend, not hardcoded */
  deviceCount: number;
  studentCount: number;
  /** Whether this school currently has any active (NEW/ACK) incidents. */
  hasActiveIncident?: boolean;
}

export const POLICE_SCHOOLS: School[] = [
  {
    id: "lincoln-high",
    name: "Lincoln High School",
    address: "1200 N Maple Ave",
    district: "District 4",
    deviceCount: 9,
    studentCount: 1840,
  },
];
