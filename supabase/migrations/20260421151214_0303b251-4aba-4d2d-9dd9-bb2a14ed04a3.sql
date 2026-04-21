-- =========================================================================
-- TacticalEye backend — initial schema
-- =========================================================================

-- ---------- ENUMS ----------
create type public.app_role as enum ('school_operator', 'police_dispatcher', 'admin');
create type public.device_type as enum ('camera', 'microphone');
create type public.device_status as enum ('online', 'warning', 'triggered', 'offline');
create type public.incident_type as enum ('Gunshot', 'Suspicious Activity', 'Fire', 'Medical', 'Other');
create type public.incident_source as enum ('AUDIO-AI', 'VIDEO-AI', 'MANUAL');
create type public.incident_status as enum ('NEW', 'ACKNOWLEDGED', 'RESOLVED');
create type public.incident_severity as enum ('Low', 'Medium', 'High', 'Critical');

-- ---------- TENANT TABLES ----------
create table public.schools (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  address text not null,
  district text not null,
  floorplan_path text,
  student_count int not null default 0,
  created_at timestamptz not null default now()
);

create table public.police_departments (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  jurisdiction text not null,
  created_at timestamptz not null default now()
);

create table public.department_schools (
  department_id uuid not null references public.police_departments(id) on delete cascade,
  school_id uuid not null references public.schools(id) on delete cascade,
  primary key (department_id, school_id)
);

-- ---------- USER ROLES ----------
create table public.user_roles (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  role public.app_role not null,
  school_id uuid references public.schools(id) on delete cascade,
  department_id uuid references public.police_departments(id) on delete cascade,
  created_at timestamptz not null default now(),
  unique (user_id, role, school_id, department_id)
);

create index on public.user_roles(user_id);

-- ---------- HELPER FUNCTIONS (security definer, no recursion) ----------
create or replace function public.has_role(_user_id uuid, _role public.app_role)
returns boolean
language sql stable security definer set search_path = public
as $$
  select exists (
    select 1 from public.user_roles
    where user_id = _user_id and role = _role
  );
$$;

create or replace function public.has_school_access(_school uuid)
returns boolean
language sql stable security definer set search_path = public
as $$
  select exists (
    select 1 from public.user_roles ur
    where ur.user_id = auth.uid() and (
         (ur.role = 'school_operator'   and ur.school_id = _school)
      or (ur.role = 'police_dispatcher' and exists (
            select 1 from public.department_schools ds
            where ds.department_id = ur.department_id
              and ds.school_id = _school))
      or  ur.role = 'admin'
    )
  );
$$;

create or replace function public.has_department_access(_department uuid)
returns boolean
language sql stable security definer set search_path = public
as $$
  select exists (
    select 1 from public.user_roles ur
    where ur.user_id = auth.uid()
      and (
        (ur.role = 'police_dispatcher' and ur.department_id = _department)
        or ur.role = 'admin'
      )
  );
$$;

-- ---------- DEVICES ----------
create table public.devices (
  id uuid primary key default gen_random_uuid(),
  school_id uuid not null references public.schools(id) on delete cascade,
  name text not null,
  type public.device_type not null,
  location text not null,
  status public.device_status not null default 'online',
  x numeric(5,2) not null,
  y numeric(5,2) not null,
  feed_url text,
  last_event text,
  last_seen timestamptz not null default now()
);

create index on public.devices(school_id);

-- ---------- INCIDENTS ----------
create table public.incidents (
  id uuid primary key default gen_random_uuid(),
  display_id text unique,
  school_id uuid not null references public.schools(id) on delete cascade,
  created_at timestamptz not null default now(),
  location text not null,
  type public.incident_type not null,
  source public.incident_source not null,
  status public.incident_status not null default 'NEW',
  severity public.incident_severity not null,
  description text,
  probability numeric(4,3),
  audio_path text,
  video_path text,
  video_confirmed boolean not null default false,
  reported_by uuid references auth.users(id) on delete set null
);

create index on public.incidents(school_id, created_at desc);
create index on public.incidents(school_id, status);

create table public.incident_events (
  id uuid primary key default gen_random_uuid(),
  incident_id uuid not null references public.incidents(id) on delete cascade,
  timestamp timestamptz not null default now(),
  label text not null,
  detail text,
  actor_id uuid references auth.users(id) on delete set null
);

create index on public.incident_events(incident_id, timestamp);

-- ---------- NOTIFICATIONS ----------
create table public.notifications (
  id uuid primary key default gen_random_uuid(),
  school_id uuid not null references public.schools(id) on delete cascade,
  incident_id uuid references public.incidents(id) on delete cascade,
  timestamp timestamptz not null default now(),
  device_type public.device_type not null,
  location text not null,
  message text not null,
  probability numeric(4,3),
  audio_path text,
  video_path text,
  source public.incident_source not null
);

create index on public.notifications(school_id, timestamp desc);

-- ---------- MESSAGES ----------
create table public.messages (
  id uuid primary key default gen_random_uuid(),
  school_id uuid not null references public.schools(id) on delete cascade,
  incident_id uuid references public.incidents(id) on delete cascade,
  sender_id uuid not null references auth.users(id) on delete cascade,
  sender_role public.app_role not null,
  timestamp timestamptz not null default now(),
  text text,
  incident_report jsonb
);

create index on public.messages(school_id, timestamp);

-- =========================================================================
-- TRIGGERS
-- =========================================================================

-- Per-school, per-day display_id like INC-YYYYMMDD-NNN
create or replace function public.gen_incident_display_id()
returns trigger language plpgsql security definer set search_path = public as $$
declare
  prefix text;
  n int;
begin
  prefix := 'INC-' || to_char(coalesce(new.created_at, now()), 'YYYYMMDD');
  select count(*) + 1 into n
  from public.incidents
  where school_id = new.school_id
    and display_id like prefix || '-%';
  new.display_id := prefix || '-' || lpad(n::text, 3, '0');
  return new;
end;
$$;

create trigger trg_incidents_display_id
before insert on public.incidents
for each row
when (new.display_id is null)
execute function public.gen_incident_display_id();

-- Append a timeline entry whenever status changes
create or replace function public.incident_status_event()
returns trigger language plpgsql security definer set search_path = public as $$
begin
  if new.status is distinct from old.status then
    insert into public.incident_events (incident_id, label, actor_id)
    values (new.id, 'Status → ' || new.status::text, auth.uid());
  end if;
  return new;
end;
$$;

create trigger trg_incidents_status_event
after update on public.incidents
for each row execute function public.incident_status_event();

-- Auto-create a notification when an AI-detected incident is inserted
create or replace function public.notify_on_incident_insert()
returns trigger language plpgsql security definer set search_path = public as $$
begin
  if new.source in ('AUDIO-AI', 'VIDEO-AI') then
    insert into public.notifications (
      school_id, incident_id, device_type, location, message,
      probability, audio_path, video_path, source
    ) values (
      new.school_id,
      new.id,
      case when new.source = 'AUDIO-AI' then 'microphone'::public.device_type
           else 'camera'::public.device_type end,
      new.location,
      new.source::text || ' detected ' || lower(new.type::text) || ' at ' || new.location,
      new.probability,
      new.audio_path,
      new.video_path,
      new.source
    );
  end if;

  -- Initial timeline entry
  insert into public.incident_events (incident_id, label, detail, actor_id)
  values (
    new.id,
    case
      when new.source = 'MANUAL' then 'Manual report filed'
      else new.source::text || ' detection'
    end,
    'Detection at ' || new.location,
    new.reported_by
  );
  return new;
end;
$$;

create trigger trg_incidents_notify
after insert on public.incidents
for each row execute function public.notify_on_incident_insert();

-- =========================================================================
-- ROW LEVEL SECURITY
-- =========================================================================

alter table public.schools enable row level security;
alter table public.police_departments enable row level security;
alter table public.department_schools enable row level security;
alter table public.user_roles enable row level security;
alter table public.devices enable row level security;
alter table public.incidents enable row level security;
alter table public.incident_events enable row level security;
alter table public.notifications enable row level security;
alter table public.messages enable row level security;

-- ---- schools ----
create policy "schools: tenant read"
  on public.schools for select to authenticated
  using (public.has_school_access(id));
create policy "schools: admin write"
  on public.schools for all to authenticated
  using (public.has_role(auth.uid(), 'admin'))
  with check (public.has_role(auth.uid(), 'admin'));

-- ---- police_departments ----
create policy "departments: own read"
  on public.police_departments for select to authenticated
  using (public.has_department_access(id));
create policy "departments: admin write"
  on public.police_departments for all to authenticated
  using (public.has_role(auth.uid(), 'admin'))
  with check (public.has_role(auth.uid(), 'admin'));

-- ---- department_schools ----
create policy "dept_schools: read by either side"
  on public.department_schools for select to authenticated
  using (public.has_department_access(department_id) or public.has_school_access(school_id));
create policy "dept_schools: admin write"
  on public.department_schools for all to authenticated
  using (public.has_role(auth.uid(), 'admin'))
  with check (public.has_role(auth.uid(), 'admin'));

-- ---- user_roles ----
create policy "user_roles: self read"
  on public.user_roles for select to authenticated
  using (user_id = auth.uid() or public.has_role(auth.uid(), 'admin'));
create policy "user_roles: admin write"
  on public.user_roles for all to authenticated
  using (public.has_role(auth.uid(), 'admin'))
  with check (public.has_role(auth.uid(), 'admin'));

-- ---- devices ----
create policy "devices: tenant read"
  on public.devices for select to authenticated
  using (public.has_school_access(school_id));
create policy "devices: tenant write"
  on public.devices for all to authenticated
  using (public.has_school_access(school_id))
  with check (public.has_school_access(school_id));

-- ---- incidents ----
create policy "incidents: tenant read"
  on public.incidents for select to authenticated
  using (public.has_school_access(school_id));
create policy "incidents: tenant insert"
  on public.incidents for insert to authenticated
  with check (public.has_school_access(school_id));
create policy "incidents: tenant update"
  on public.incidents for update to authenticated
  using (public.has_school_access(school_id))
  with check (public.has_school_access(school_id));

-- ---- incident_events ----
create policy "events: tenant read"
  on public.incident_events for select to authenticated
  using (exists (
    select 1 from public.incidents i
    where i.id = incident_events.incident_id and public.has_school_access(i.school_id)
  ));
create policy "events: tenant insert"
  on public.incident_events for insert to authenticated
  with check (exists (
    select 1 from public.incidents i
    where i.id = incident_events.incident_id and public.has_school_access(i.school_id)
  ));

-- ---- notifications ----
create policy "notifications: tenant read"
  on public.notifications for select to authenticated
  using (public.has_school_access(school_id));
create policy "notifications: tenant insert"
  on public.notifications for insert to authenticated
  with check (public.has_school_access(school_id));

-- ---- messages ----
create policy "messages: tenant read"
  on public.messages for select to authenticated
  using (public.has_school_access(school_id));
create policy "messages: send as self"
  on public.messages for insert to authenticated
  with check (sender_id = auth.uid() and public.has_school_access(school_id));

-- =========================================================================
-- REALTIME
-- =========================================================================
alter publication supabase_realtime add table public.incidents;
alter publication supabase_realtime add table public.incident_events;
alter publication supabase_realtime add table public.notifications;
alter publication supabase_realtime add table public.messages;
alter publication supabase_realtime add table public.devices;

alter table public.incidents replica identity full;
alter table public.incident_events replica identity full;
alter table public.notifications replica identity full;
alter table public.messages replica identity full;
alter table public.devices replica identity full;

-- =========================================================================
-- STORAGE BUCKETS (private)
-- =========================================================================
insert into storage.buckets (id, name, public)
values ('media', 'media', false), ('floorplans', 'floorplans', false)
on conflict (id) do nothing;

-- Tenant read for media: object path prefix is "{school_id}/..."
create policy "media: tenant read"
  on storage.objects for select to authenticated
  using (
    bucket_id = 'media'
    and public.has_school_access((storage.foldername(name))[1]::uuid)
  );
create policy "media: tenant write"
  on storage.objects for insert to authenticated
  with check (
    bucket_id = 'media'
    and public.has_school_access((storage.foldername(name))[1]::uuid)
  );

create policy "floorplans: tenant read"
  on storage.objects for select to authenticated
  using (
    bucket_id = 'floorplans'
    and public.has_school_access((storage.foldername(name))[1]::uuid)
  );
create policy "floorplans: admin write"
  on storage.objects for insert to authenticated
  with check (
    bucket_id = 'floorplans'
    and public.has_role(auth.uid(), 'admin')
  );